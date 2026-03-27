import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    result=[]
    if max_len is None:
        max_len=max(len(seq) for seq in seqs)
    for seq in seqs:
        if len(seq)>max_len:
            seq=seq[:max_len]
        else:
            while(len(seq)<max_len):
                seq.append(pad_value)
        result.append(seq)
    return result
     