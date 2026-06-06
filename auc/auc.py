import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr=np.array(fpr)
    tpr=np.array(tpr)
    n=len(fpr)
    auc=0
    for i in range(0,n-1):
        auc+=1/2*(tpr[i]+tpr[i+1])*(fpr[i+1]-fpr[i])
    return auc
    pass