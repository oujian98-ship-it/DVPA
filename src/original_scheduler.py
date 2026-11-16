"""Algorithm 1 epsilon-remainder recurrence, with numeric RDP and T-t+1 rounds.

The paper's literal T-t is undefined at the last round. The full RDP ledger
is checked separately before each release; epsilon subtraction is conservative.
"""
from dataclasses import dataclass
import numpy as np
from .accountant import calibrate, calibrate_remaining, increment, epsilon, ORDERS

@dataclass
class RoundParams:
    q_n: np.ndarray
    sigma_n: np.ndarray
    clip_n: np.ndarray
    q_bar: float
    sigma_global: float

def groups(n, proportions):
    p = np.asarray(proportions, dtype=float)
    if (p <= 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError('Privacy proportions must be positive and sum to one')
    counts = np.floor(n*p).astype(int)
    for j in np.argsort(-(n*p-counts))[:n-counts.sum()]:
        counts[j] += 1
    return np.repeat(np.arange(len(p)), counts)

class PrivacyScheduler:
    def __init__(self, config):
        p, tr = config['privacy'], config['train']
        n, rounds = config['data']['clients'], tr['rounds']
        method = config['method']['temporal']
        if method not in {'paper_spend_as_you_go', 'fedavg', 'dp_fedavg', 'idp_fedavg'}:
            raise ValueError('Unknown temporal method')
        ids = groups(n, p['group_proportions'])
        targets = np.array(p['privacy_groups'])[ids].astype(float)
        if method == 'dp_fedavg':
            targets[:] = targets.min()
        self.targets, self.params_by_round = targets, []
        calibration = p.get('calibration', 'paper_epsilon_remainder' if method == 'paper_spend_as_you_go' else 'full_rdp_remaining')
        if calibration not in {'paper_epsilon_remainder', 'full_rdp_remaining'}:
            raise ValueError('Unknown privacy calibration')
        ledger = np.zeros((len(p['privacy_groups']), len(ORDERS)))
        for t in range(1, rounds+1):
            qs, ss = [], []
            for g in range(len(p['privacy_groups'])):
                target = min(p['privacy_groups']) if method == 'dp_fedavg' else p['privacy_groups'][g]
                q = p['q_save'][g] if method == 'paper_spend_as_you_go' and t < p['transition_round'] else p['q_spend']
                if not 0 < q <= 1:
                    raise ValueError('Sampling probability must be in (0,1]')
                if method == 'fedavg':
                    q, sigma = 1.0, 0.0
                else:
                    if method in {'dp_fedavg', 'idp_fedavg'}:
                        # Uniform baselines calibrate once for the entire
                        # horizon. Epsilon subtraction spuriously increases
                        # sigma over time and is not uniform privacy spending.
                        sigma = calibrate(target, p['delta'], p['q_spend'], rounds)
                    elif calibration == 'full_rdp_remaining':
                        sigma = calibrate_remaining(target, p['delta'], p['q_spend'], rounds-t+1, ledger[g])
                    else:
                        remaining = target-epsilon(ledger[g], p['delta'])
                        sigma = calibrate(remaining, p['delta'], p['q_spend'], rounds-t+1)
                    ledger[g] += increment(q, sigma)
                qs.append(q)
                ss.append(sigma)
            qn, sn = np.array(qs)[ids], np.array(ss)[ids]
            sg = 0.0 if method == 'fedavg' else float(1/np.mean(1/sn))
            cn = np.full(n, p['average_clip_c']) if method == 'fedavg' else p['average_clip_c']*sg/sn
            self.params_by_round.append(RoundParams(qn, sn, cn, float(qn.mean()), sg))

    def params(self, t):
        return self.params_by_round[t-1]
