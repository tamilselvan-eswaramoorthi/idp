import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from model import CRAFT
from utils.getboxes import getDetBoxes
from utils.grouper import group_text_box
from utils.general import diff, copyStateDict, adjustResultCoordinates, draw_detections
from utils.imgproc import cvt2HeatmapImg, resize_aspect_ratio, normalizeMeanVariance, loadImage


def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    return boxes_list, polys_list, ret_score_text

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net

def get_textbox(detector, image, min_size = 20, canvas_size=1280, mag_ratio=1.5, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False, device='cpu', optimal_num_chars=None, **kwargs):
    text_box_list = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list, score_text = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars)

    # cv2.imwrite('temp.png', score_text)

    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        text_box_list.append(single_img_result)

    horizontal_list_agg, free_list_agg = [], []
    for text_box in text_box_list:
        horizontal_list, free_list = group_text_box(text_box, sort_output = (optimal_num_chars is None))
        if min_size:
            horizontal_list = [i for i in horizontal_list if max(
                i[1] - i[0], i[3] - i[2]) > min_size]
            free_list = [i for i in free_list if max(
                diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
        horizontal_list_agg.append(horizontal_list)
        free_list_agg.append(free_list)


    return horizontal_list_agg, free_list_agg

def initDetector(detector_path):
    return get_detector(detector_path)

if __name__ == "__main__":
    model = initDetector("weights/craft_mlt_25k.pth")
    image = loadImage('sample.png')
    horizontal_list_agg, free_list_agg = get_textbox(model, image)
    results = draw_detections([image], horizontal_list_agg, free_list_agg)
    cv2.imwrite('temp.png', results[0])