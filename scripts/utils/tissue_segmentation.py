import os, sys
import numpy as np
import openslide
import PIL
from PIL import Image
import cv2

from skimage import img_as_ubyte


def auto_adjust_image(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Adjusts contrast and brightness of a numpy array image using CLAHE.
    
    Args:
        image (numpy.ndarray): Input image (BGR or Grayscale).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        
    Returns:
        adjusted_image (numpy.ndarray): The enhanced image.
    """
    # Check if image is color (3 channels) or grayscale
    if len(image.shape) == 3:
        # Convert to LAB color space
        # L = Lightness, A/B = color dimensions
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)

        # Merge back and convert to BGR
        limg = cv2.merge((cl, a, b))
        adjusted_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images, apply CLAHE directly
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        adjusted_image = clahe.apply(image)

    return adjusted_image

### Otsu Segmentation is on 1024 thumbnail
def Otsu_Seg(Slide):
    
    Img = Slide.get_thumbnail((1024, 1024)) ## when you read thumbnail
    Img = np.asarray(Img.convert("RGB"))
    
    gray_Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    gray_Img = cv2.GaussianBlur(gray_Img, (5,5), 0)
    ret, seg = cv2.threshold(gray_Img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seg = (255 - seg)
    
    return Img, seg