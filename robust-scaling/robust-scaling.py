import numpy as np
def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    # Write code here
    values=np.array(values)
    if len(values)==1:
        return [0.0]
    sort=np.sort(values)
    mid=len(values)//2
    if len(values)%2==0:
        median=np.median(sort)
    else:
        median=sort[mid]
    if len(values)%2==0:
        Q1 = sort[:mid]
        Q3 = sort[mid:]
    else:
        Q1= sort[:mid]
        Q3=sort[mid+1:]
    Q1=np.median(Q1)
    Q3=np.median(Q3)
    IQR=Q3-Q1
    if IQR==0:
        return values-median
    x_scaled=(values-median)/IQR
    return x_scaled