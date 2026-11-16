"""Pure offline utilities. Never imported by the private training loop."""
import numpy as np

def conditional_mse(ideal, clipped, q, scale):
    ideal, clipped, q = np.asarray(ideal), np.asarray(clipped), np.asarray(q)
    n, d = clipped.shape
    denominator = q.sum()
    bias = (q[:, None]*clipped).sum(0)/denominator-ideal.mean(0)
    sampling = (q*(1-q)*(clipped**2).sum(1)).sum()/denominator**2
    return float(bias@bias + sampling + d*scale**2/denominator**2)

def decision_metrics(predicted, truth):
    from .p4_metrics import decision_metrics as evaluate
    return evaluate(predicted, truth)
