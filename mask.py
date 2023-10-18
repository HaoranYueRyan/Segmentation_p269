import numpy as np
import napari
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt

from stardist.models import StarDist2D
from csbdeep.utils import normalize
model = StarDist2D.from_pretrained('2D_versatile_fluo')

def save_fig(fig_id, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(D_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def Stardist_Segmentation(image):
    """
    Perform Stardist Segmentation,
    :param image: Image to Segment
    :return: Segmented Objects and Labels
    """
    label_objects, nb_labels = model.predict_instances(normalize(image))
    cleared = remove_small_objects(clear_border(label_objects), 10)
    segmented_cells, cell_number = ndi.label(cleared)
    return segmented_cells, cell_number

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect

def Napari_Display(image, segmented_cells, cell_number):
    """

    Display Image in Napari Viewer. Segmented cells and rectangles around segmented cells
   :param image: Image, Segmented Cells, and Cell Number
    """
    properties = measure.regionprops_table(
        segmented_cells, properties=('label', 'bbox')
    )
    # create the bounding box rectangles
    bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])

    viewer=napari.Viewer()
    viewer.add_image(image)
    viewer.add_labels(segmented_cells, name=f'segmentation {cell_number} of cells')
    viewer.add_shapes(
        bbox_rects,
        face_color='transparent',
        edge_color='orange',
        properties=properties,
        name='bounding box',
    )


def get_features(label_image, image, featurelist):
    """
    Generate dataframe with single cell data of selected features for segmented image
    :param label_image: segmented imageg from watershed
    :param image: original image
    :param featurelist: list of measured features e.g.['area','max_intensity','mean_intensity']
    :return: dataframe with single cell features
    """
    props = measure.regionprops_table(label_image, image, properties=featurelist)
    data = pd.DataFrame(props)
    return data

def Stardist_Counting(image):
    if image.max()>200:
        label_objects, nb_labels = model.predict_instances(normalize(image))
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > 100
        mask_sizes[0] = 0
        cells_cleaned = mask_sizes[label_objects]
        segmented_cells_, cell_number = ndi.label(cells_cleaned)
        props = measure.regionprops_table(segmented_cells_, image,properties=['area',
                                                                              'mean_intensity'])
        df=pd.DataFrame(props)
        df['integrated_intensity']=df['area']*df['mean_intensity']
        return df, cell_number
    else:
        return pd.DataFrame(), 0
model = StarDist2D.from_pretrained('2D_versatile_fluo')
