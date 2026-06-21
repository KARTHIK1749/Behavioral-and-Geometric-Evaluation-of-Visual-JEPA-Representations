
import numpy as np

def entropy(probs):
    return float(-np.sum(probs * np.log(probs + 1e-9)))
