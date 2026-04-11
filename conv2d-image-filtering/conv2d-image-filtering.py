import numpy as np
def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    image=np.array(image)
    kernel=np.array(kernel)
    h,w=image.shape
    f=kernel.shape[0]
    image=np.pad(image,pad_width=((padding,padding),(padding,padding)),mode='constant',constant_values=0)
    h_out=(h+2*padding-f)//stride+1
    w_out=(w+2*padding-f)//stride+1
    op=np.zeros((h_out,w_out))
    for i in range(h_out):
        for j in range(w_out):
            out=image[i*stride:i*stride+f,j*stride:j*stride+f]
            op[i,j]=np.sum(out*kernel)

    return op.tolist()