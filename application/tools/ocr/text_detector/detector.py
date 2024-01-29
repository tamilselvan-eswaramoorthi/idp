import cv2
import math
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from .model import CRAFT
from .utils.load_image import loadImage
from .utils.getboxes import getDetBoxes
from .utils.grouper import group_text_box
from .utils.imgproc import cvt2HeatmapImg, resize_aspect_ratio, normalizeMeanVariance
from .utils.general import diff, copyStateDict, adjustResultCoordinates, compute_ratio_and_resize, calculate_ratio, four_point_transform

class Detector:

    def __init__(self, detector_path, min_size = 20, canvas_size=1280, mag_ratio=1.5) -> None:
      self.device = 'cpu'
      self.model = self.get_detector(detector_path)

      self.min_size = min_size
      self.canvas_size = canvas_size
      self.mag_ratio = mag_ratio


    def test_net(self, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False, estimate_num_chars=False):
        if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
            image_arrs = image
        else:                                                        # image is single numpy array
            image_arrs = [image]

        img_resized_list = []
        # resize
        for img in image_arrs:
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, self.canvas_size,
                                                                        interpolation=cv2.INTER_LINEAR,
                                                                        mag_ratio=self.mag_ratio)
            img_resized_list.append(img_resized)
        ratio_h = ratio_w = 1 / target_ratio
        # preprocessing
        x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
            for n_img in img_resized_list]
        x = torch.from_numpy(np.array(x))
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

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

    def get_detector(self, trained_model, quantize=True, cudnn_benchmark=False):
        net = CRAFT()

        if self.device == 'cpu':
            net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=self.device)))
            if quantize:
                try:
                    torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
                except:
                    pass
        else:
            net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=self.device)))
            net = torch.nn.DataParallel(net).to(self.device)
            cudnn.benchmark = cudnn_benchmark

        net.eval()
        return net

    def get_textbox(self, image,  optimal_num_chars=None):
        text_box_list = []
        estimate_num_chars = optimal_num_chars is not None
        bboxes_list, polys_list, score_text = self.test_net(image, estimate_num_chars = estimate_num_chars)

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
            if self.min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > self.min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > self.min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)


        return horizontal_list_agg[0], free_list_agg[0]

    def get_image_list(self, horizontal_list, free_list, img, model_height = 64, sort_output = True):
        image_list = []
        maximum_y,maximum_x = img.shape

        max_ratio_hori, max_ratio_free = 1,1
        for box in free_list:
            rect = np.array(box, dtype = "float32")
            transformed_img = four_point_transform(img, rect)
            ratio = calculate_ratio(transformed_img.shape[1],transformed_img.shape[0])
            new_width = int(model_height*ratio)
            if new_width == 0:
                pass
            else:
                crop_img,ratio = compute_ratio_and_resize(transformed_img,transformed_img.shape[1],transformed_img.shape[0],model_height)
                image_list.append( (box,crop_img) ) # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                max_ratio_free = max(ratio, max_ratio_free)


        max_ratio_free = math.ceil(max_ratio_free)

        for box in horizontal_list:
            x_min = max(0,box[0])
            x_max = min(box[1],maximum_x)
            y_min = max(0,box[2])
            y_max = min(box[3],maximum_y)
            crop_img = img[y_min : y_max, x_min:x_max]
            width = x_max - x_min
            height = y_max - y_min
            ratio = calculate_ratio(width,height)
            new_width = int(model_height*ratio)
            if new_width == 0:
                pass
            else:
                crop_img,ratio = compute_ratio_and_resize(crop_img,width,height,model_height)
                image_list.append( ( [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]] ,crop_img) )
                max_ratio_hori = max(ratio, max_ratio_hori)

        max_ratio_hori = math.ceil(max_ratio_hori)
        max_ratio = max(max_ratio_hori, max_ratio_free)
        max_width = math.ceil(max_ratio)*model_height

        if sort_output:
            image_list = sorted(image_list, key=lambda item: item[0][0][1]) # sort by vertical position
        return image_list, max_width
