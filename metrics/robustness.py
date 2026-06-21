
import numpy as np

def robustness_curve(scores):
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }
