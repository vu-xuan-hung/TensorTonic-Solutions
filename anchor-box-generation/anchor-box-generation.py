import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride= image_size/feature_size
    anchorbox=[]
    scales=np.array(scales)
    aspect_ratios=np.array(aspect_ratios)
    for i in range(feature_size):
        for j in range(feature_size):
            cx=(j+0.5)*stride
            cy=(i+0.5)*stride
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    w=scale*np.sqrt(aspect_ratio)
                    h=scale/np.sqrt(aspect_ratio)
                    anchorbox.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return anchorbox
        