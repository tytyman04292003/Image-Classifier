from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    
    #Opens an image
    image = Image.open(image_path)
    width_now, height_now = image.size
    #Process edge-croppings
    if width_now < height_now:
        new_height = int(height_now * 256 / width_now)
        image = image.resize((256, new_height))
    else:
        new_width = int(width_now *256 / height_now)
        image = image.resize((new_width, 256))
    #Crop out center of image
    width, height = image.size # Get dimensions
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    #Define and implement color channels
    np_image = np.array(image)
    np_image = np_image/255 
    #Standardize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    processed_image = (np_image - mean)/std
    #Transpose colors
    transposed_image = processed_image.transpose((2,0,1)) 
    
    return transposed_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plot.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
