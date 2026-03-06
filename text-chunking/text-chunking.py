import numpy as np
def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    tokens=np.array(tokens)
    step=chunk_size-overlap
    chunks=[]
    for i in range(0,len(tokens),step):
        chunks.append(tokens[i:i+chunk_size].tolist())
        if(i+chunk_size>=len(tokens)):
            break
    return chunks
            
        