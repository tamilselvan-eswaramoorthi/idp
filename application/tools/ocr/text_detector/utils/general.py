import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
from .getboxes import getDetBoxes
from .imgproc import cvt2HeatmapImg

def calculate_ratio(width,height):
    '''
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = 1./ratio
    return ratio

def compute_ratio_and_resize(img,width,height,model_height):
    '''
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = calculate_ratio(width,height)
        img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.Resampling.LANCZOS)
    else:
        img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.Resampling.LANCZOS)
    return img,ratio

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def diff(input_list):
    return max(input_list)-min(input_list)

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

def draw_detections(image, horizontal_list, free_list, color = (0, 0, 255), thickness = 1):
    maximum_y,maximum_x, _ = image.shape
    for box in horizontal_list:
        x_min = max(0,box[0])
        x_max = min(box[1],maximum_x)
        y_min = max(0,box[2])
        y_max = min(box[3],maximum_y)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness )

    for box in free_list:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        image = cv2.polylines(image, [box], 1, color, thickness) 
    return image

def save_outputs(image, region_scores, affinity_scores, text_threshold, link_threshold,
                                           low_text, outoput_path, confidence_mask = None):
    """save image, region_scores, and affinity_scores in a single image. region_scores and affinity_scores must be
    cpu numpy arrays. You can convert GPU Tensors to CPU numpy arrays like this:
    >>> array = tensor.cpu().data.numpy()
    When saving outputs of the network during training, make sure you convert ALL tensors (image, region_score,
    affinity_score) to numpy array first.
    :param image: numpy array
    :param region_scores: [] 2D numpy array with each element between 0~1.
    :param affinity_scores: same as region_scores
    :param text_threshold: 0 ~ 1. Closer to 0, characters with lower confidence will also be considered a word and be boxed
    :param link_threshold: 0 ~ 1. Closer to 0, links with lower confidence will also be considered a word and be boxed
    :param low_text: 0 ~ 1. Closer to 0, boxes will be more loosely drawn.
    :param outoput_path:
    :param confidence_mask:
    :return:
    """

    assert region_scores.shape == affinity_scores.shape
    assert len(image.shape) - 1 == len(region_scores.shape)

    boxes, polys = getDetBoxes(region_scores, affinity_scores, text_threshold, link_threshold,
                                           low_text, False)
    boxes = np.array(boxes, np.int32) * 2
    if len(boxes) > 0:
        np.clip(boxes[:, :, 0], 0, image.shape[1])
        np.clip(boxes[:, :, 1], 0, image.shape[0])
        for box in boxes:
            cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))

    target_gaussian_heatmap_color = cvt2HeatmapImg(region_scores)
    target_gaussian_affinity_heatmap_color = cvt2HeatmapImg(affinity_scores)

    if confidence_mask is not None:
        confidence_mask_gray = cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray], axis=0)
        output = np.hstack([image, output])

    else:
        gt_scores = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=0)
        output = np.hstack([image, gt_scores])

    cv2.imwrite(outoput_path, output)
    return output
