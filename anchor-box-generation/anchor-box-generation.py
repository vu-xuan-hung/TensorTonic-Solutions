import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride=image_size/feature_size
    num_anchor=feature_size**2*len(aspect_ratios)

    anchor=[]
    for i in range(feature_size):
        for j in range(feature_size):
            cx=(j+0.5)*stride
            cy=(i+0.5)*stride
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    w=scale*np.sqrt(aspect_ratio)
                    h=scale/np.sqrt(aspect_ratio)
                    anchor.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return anchor