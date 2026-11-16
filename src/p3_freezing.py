"""Author-runner adapter for historical DP aggregate feedback."""
from .p1_freezing import StaticMask
from .dp_feedback_freezer import FeedbackFreezer, DEFAULTS, validate_feedback


def validate_feedback_run(config):
    adaptive=config.get('mode')=='ADAPTIVE'
    if adaptive!=('feedback' in config):raise ValueError('ADAPTIVE mode requires feedback and vice versa')
    if not adaptive:
        if 'diagnostic_freezing' in config:raise ValueError('Diagnostic policy requires ADAPTIVE adapter')
        return
    feedback=config['feedback']
    if not isinstance(feedback,dict) or set(feedback)-set(DEFAULTS):
        raise ValueError('Unknown feedback fields')
    validate_feedback(feedback)
    if not config.get('privacy_enabled',True):raise ValueError('P3 feedback requires DP aggregate releases')
    if config.get('freeze_rounds'):raise ValueError('P3 cannot mix static freeze events with feedback')
    if 'diagnostic_freezing' in config:
        from .diagnostic_freezing import validate_diagnostic
        validate_diagnostic(config['diagnostic_freezing'],feedback,config['rounds'])
    if config.get('validation_protocol'):
        import json
        from pathlib import Path
        try:kind=json.loads(Path(config['validation_protocol']).read_text(encoding='utf-8')).get('kind')
        except (OSError,ValueError) as exc:raise ValueError('Invalid feedback validation protocol') from exc
        allowed=({'controlled_freezing_diagnostics_v1','controlled_joint_allocation_freezing_v1'} if 'diagnostic_freezing' in config else {'controlled_feedback_comparison_v1'})
        if kind not in allowed:raise ValueError('Static P1 protocols cannot authorize adaptive P3')


class AdaptiveMask(StaticMask):
    def __init__(self, model, feedback, seed=42, diagnostic=None, rounds=None):
        super().__init__(model,'NF',[],seed)
        self.mode='ADAPTIVE'
        if diagnostic is None:self.controller=FeedbackFreezer(self.names,feedback,seed)
        else:
            from .diagnostic_freezing import DiagnosticFreezer
            self.controller=DiagnosticFreezer(self.names,self.dims,feedback,diagnostic,rounds)

    def decide(self, t):
        active=self.controller.decide(t)
        masks={n:mask.clone().fill_(active[n.split('.')[0]]) for n,mask in self.previous.items()}
        self.previous={n:m.clone() for n,m in masks.items()};self.last_round=t
        return masks

    def observe(self, t, released, variance):
        if set(released)!=set(self.shapes) or any(tuple(v.shape)!=self.shapes[n] for n,v in released.items()):
            raise ValueError('Released parameter registry/shape mismatch')
        # Group weight+bias without concatenation; only active published coordinates.
        groups={g:tuple(v for n,v in released.items() if n.split('.')[0]==g)
                for g in self.names if self.controller.active[g]}
        return self.controller.observe(t,groups,variance)


def feedback_preview(config):
    """Validate configuration/registry only. Adaptive trajectories need releases."""
    from .upstream_runner import source_model
    from .p2_allocation import validate_allocation, weights
    validate_feedback_run(config)
    if config.get('mode')!='ADAPTIVE':raise ValueError('P3 preview requires ADAPTIVE mode')
    allocation=config.get('allocation',{'layer_clip':'equal_active'})
    validate_allocation(allocation)
    model=source_model()('fmnist',config['batch_size'])
    mask=AdaptiveMask(model,config['feedback'],config['seed'])
    initial_weights=weights(mask.names,mask.dims,dict.fromkeys(mask.names,True),1,allocation)
    return dict(stage='P3',scope='configuration only; no training, data access or fabricated DP history',
                feedback=mask.controller.config,dimensions=mask.dims,
                protected_groups=[g for g in mask.names if g in mask.controller.protected],
                initial_weights=initial_weights,adaptive_freeze_rounds=None,
                decision_timing='observe released round t; apply at round t+1 or later')
