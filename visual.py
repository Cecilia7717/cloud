import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from qonnx import util
import rasterio
from skimage.transform import resize


def read_tiff(filepath: str) -> np.ndarray:
    """Return image in channel-last format (HWC).

    Args:
        filepath (str): Path to read image from.

    Returns:
        np.ndarray: Image in HWC
    """
    with rasterio.open(filepath, mode="r", crs=None, transform=None) as i_raster:
        
        return np.array(i_raster.read()).astype(np.float32)
    
def read_image(img_folder,image_relative_path):
    """Read image

    Args:
        img_folder (String): Image/DB base folder
        image_relative_path (String): Relative path of the image inside the Image/DB base folder

    Returns:
        np.array: image readed
    """
    # Get the path to the image
    img_path = os.path.join(img_folder, image_relative_path)
    return read_tiff(img_path)
    
def postprocess(pred, ch_last=False):
    """Apply post-processing

    Args:
        pred (np.array): Prediction of the model

    Returns:
        np.array: argmax of the pred
    """
    if isinstance(pred, list): pred = pred[0]
    if ch_last:
        return np.argmax(pred,axis=-1)
    else:
        return np.argmax(pred,axis=1)

def preprocess(image, ch_last=False):
    """Apply preprocessing on image

    Args:
        image (np.array): input imgae to be pre-processed

    Returns:
        np.array: Pre-processed image
    """
    # DB specific values - HARDCODED
    image_mean = 0
    image_scale = 1
    range_max = 8657
    
    image = np.clip(((image/range_max)-image_mean)/image_scale, 0.0, 1.0)
    image = (np.round(image * 2**4) * 2**-4).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    if ch_last:
        image = np.transpose(image, (0, 2, 3, 1))
    return image

def read_GT(img_folder,image_relative_path):
    """Read Ground Truth

    Args:
        img_folder (String): Image/DB base folder
        image_relative_path (String): Relative path of the image inside the Image/DB base folder

    Returns:
        np.array: image readed
    """
    # Get the path to the image
    GT_relative_path = image_relative_path.replace("Input","Masks").replace(".tif",".tiff")
    GT_path = os.path.join(img_folder, GT_relative_path)
    return read_tiff(GT_path)

import rasterio

def get_tif_using_opencv(filepath: str) -> np.ndarray:
    image = read_tiff(filepath)

    meta = {
        'driver': 'GTiff',
        'count': image.shape[0],  
        'dtype': 'float32',  
        'width': image.shape[2],
        'height': image.shape[1],
        'crs': None,  
        'transform': None,  
    }
    
    # if image.shape[0] == 1: image = image.squeeze()
    with rasterio.open("image.tif", 'w', **meta) as dst:
        dst.write(image)

    model_inputs = cv2.imread("image.tif", flags=(cv2.IMREAD_UNCHANGED))
    model_inputs = cv2.cvtColor(model_inputs, cv2.COLOR_BGR2RGB)    
    model_inputs = model_inputs.astype(np.float32)

    import os
    os.system('rm image.tif')

    return model_inputs

def write_image(path, image, do_preprocess=False):
    if do_preprocess: image = preprocess(image)
    plt.imsave(path,image)

def write_mask_overlapped(path, image, mask, alpha=0.5): # to be tested

    mask = preprocess(mask)
    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)

    yellow_overlay = np.zeros_like(image, dtype=np.float32)
    yellow_overlay[:, :, 0] = 0
    yellow_overlay[:, :, 1] = 255
    yellow_overlay[:, :, 2] = 255

    # mask_3channel = np.stack([mask]*3, axis=-1)

    blended = np.where(mask == 1, cv2.addWeighted(image, 1 - alpha, yellow_overlay, alpha, 0), image)

    plt.imsave(path, preprocess(blended).squeeze())

img_folder = "/Users/cc/Documents/cloud/"
img_rel_path = "image.tif"

image = read_image(img_folder, img_rel_path)   # returns np.ndarray
print(image.shape)