import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)      
    y_pred = np.asarray(y_pred, dtype=float)      
    N = len(y_true)          
    picked = y_pred[np.arange(N), y_true]         
    losses = -np.log(picked)                     
    return float(np.mean(losses))                
    