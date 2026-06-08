import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)
    tp=np.sum(y_pred==y_true)
    class_=np.unique(y_true)
    fp=fn=0
    for c in class_:
        fp+=np.sum((y_pred==c)&(y_true!=c))
        fn+=np.sum((y_pred!=c)&(y_true==c))
    return 2*tp/(2*tp+fp+fn)
    