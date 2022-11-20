import matplotlib.pyplot as plt
import numpy as np

#//////////////////////// DATA LADER and UTILITY FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        #print(image.shape)
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
# helper function for creating RGB of ith mask
def MasktoRGB(mask,color):
    r = np.expand_dims((mask*color[0]),axis=-1)
    g = np.expand_dims((mask*color[1]),axis=-1)
    b = np.expand_dims((mask*color[2]),axis=-1)
    img = np.concatenate((r,g,b), axis=-1)
    return img.astype(int)
