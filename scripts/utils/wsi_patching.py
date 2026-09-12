import os, sys
import numpy as np
import openslide
import PIL
from PIL import Image
import cv2

from skimage import img_as_ubyte


def patch_coord_scale(coordinates, scale):

    return np.round(coordinates * scale).astype(int)

def Visualize_Patches(Img, Patches, Patch_Size):
    Img = np.array(Img)
    for patch_id, patch_coord in enumerate(Patches):
        x, y = patch_coord[0], patch_coord[1]
        Img = cv2.rectangle(Img, (y, x), (y+Patch_Size, x+Patch_Size), (0, 0, 0), 2)
    return Img

def WSI_Patching(mask, patch_size, iou_threshold, Tissue_area):
    mask = np.array(mask)
    i_h, i_w= mask.shape[0], mask.shape[1]
    p_h, p_w = patch_size, patch_size
    threshold = iou_threshold
    area_thresh = Tissue_area
    valid_patches = []
    total_patches = 0
    for x_cord in np.arange(0, i_h, int(p_h * threshold)):
        for y_cord in np.arange(0, i_w, int(p_w * threshold)):
            if (x_cord + p_h <= i_h and y_cord + p_w < i_w):
                if np.mean(mask[x_cord:x_cord+p_h, y_cord:y_cord+p_w]) >= threshold:
                    patch_contours, patch_hierarchy = cv2.findContours(mask[x_cord:x_cord+p_h, y_cord:y_cord+p_w], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    patch_Area = 0
                    for C in patch_contours:
                        T_Area = cv2.contourArea(C)
                        patch_Area = patch_Area + T_Area
                        
                    patch_Ratio = (patch_Area/(p_h*p_w))*100
                        
                    if patch_Ratio >= area_thresh:
                        total_patches += 1
                        valid_patches.append([x_cord, y_cord])
    valid_patches = np.array(valid_patches)
    # print("Total Patches: ", total_patches)
    return valid_patches