from skimage import measure,exposure
import numpy as np
import numpy as np

import skimage
from cellpose import models
from skimage import measure, io
import cv2
import os


from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.filters import sobel
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from scipy import ndimage as ndi

def scale_img(img: np.array, percentile: tuple[float, float] = (2, 98)) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))

def normlize_img(image: np.array) -> np.array:
    mean = np.mean(image)
    std_dev = np.std(image)
    normalized_image = (image - mean) / std_dev
    return normalized_image


def find_edges(img:np.array)-> np.array:
    """find the edge of each objects"""
    edged_img=sobel(img)
    return edged_img

def Stardist_Segmentation(image):
    """
    Perform Stardist Segmentation,
    :param image: Image to Segment
    :return: Segmented Objects and Labels
    """
    model = StarDist2D.from_pretrained('2D_versatile_fluo',)
    label_objects, nb_labels = model.predict_instances(normalize(image))
    print(nb_labels)
    # cleared = remove_small_objects(clear_border(label_objects), 1)
    # segmented_cells, cell_number = ndi.label(cleared)
    return label_objects, nb_labels


def filter_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    removes border objects and filters large abd small objects from segmentation mask
    :param mask: unfiltered segmentation mask
    :return: filtered segmentation mask
    """
    cleared = clear_border(mask)
    sizes = np.bincount(cleared.ravel())
    mask_sizes = (sizes > 10)
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask

def Stardist_Counting(image):
    # if image.max()>200:
    label_objects, nb_labels = model.predict_instances(normalize(image))
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 100
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[label_objects]
    segmented_cells_, cell_number = ndi.label(cells_cleaned)

    return segmented_cells_,cell_number
        # props = measure.regionprops_table(segmented_cells_, image,properties=['area',
        #                                                                       'mean_intensity'])
        # df=pd.DataFrame(props)
        # df['integrated_intensity']=df['area']*df['mean_intensity']
    #     return df, cell_number
    # else:
    #     return pd.DataFrame(), 0

def cellpose_segmentation(image):
    """perform cellpose segmentation using nuclear mask """
    # model = models.CellposeModel(gpu=True, model_type=os.path.dirname(os.getcwd())+'/data/CellPose_models/'+Defaults.MODEL_DICT['nuclei'])
    model=models.Cellpose(model_type='nuclei')
    n_channels = [[0, 0]]
    n_mask_array, n_flows, n_styles, n_diams= model.eval(image, channels=n_channels,diameter=10,flow_threshold=0.4)
        # return cleaned up mask using filter function
        # return filter_segmentation(n_mask_array)
    segmented_cells, cell_number = ndi.label(n_mask_array)
    print(cell_number)
    return n_mask_array

# prints a list of available models
print(StarDist2D.from_pretrained())
