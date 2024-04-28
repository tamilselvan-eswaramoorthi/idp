# -*- coding: utf-8 -*-

import os

import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from text_detector.model import CRAFT
from text_detector.utils.general import copyStateDict
from text_detector.trainer.config.load_config import load_yaml, DotDict
from text_detector.trainer.metrics.eval_det_iou import DetectionIoUEvaluator
from text_detector.trainer.utils.inference_boxes import test_net, load_custom_data



def save_result_synth(img_file, img, pre_output, pre_box, gt_box=None, result_dir=""):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).astype(np.int32).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_2015(img_file, img, pre_output, pre_box, gt_box, result_dir):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    if gt_box is not None:
        for j in range(len(gt_box)):
            _gt_box = np.array(gt_box[j]["points"]).reshape(-1, 2).astype(np.int32)
            if gt_box[j]["text"] == "###":
                cv2.polylines(img, [_gt_box], True, color=(128, 128, 128), thickness=2)
            else:
                cv2.polylines(img, [_gt_box], True, color=(0, 0, 255), thickness=2)

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_2013(img_file, img, pre_output, pre_box, gt_box=None, result_dir=""):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def overlay(image, region, affinity, single_img_bbox):

    height, width, channel = image.shape

    region_score = cv2.resize(region, (width, height))
    affinity_score = cv2.resize(affinity, (width, height))

    overlay_region = cv2.addWeighted(image.copy(), 0.4, region_score, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, affinity_score, 0.6, 5)

    boxed_img = image.copy()
    for word_box in single_img_bbox:
        cv2.polylines(
            boxed_img,
            [word_box.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(0, 255, 0),
            thickness=3,
        )

    temp1 = np.hstack([image, boxed_img])
    temp2 = np.hstack([overlay_region, overlay_aff])
    temp3 = np.vstack([temp1, temp2])

    return temp3

def main_eval(model_path, backbone, data_path, evaluator, result_dir, model, mode, device):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    total_imgs_bboxes_gt, total_imgs_path = load_custom_data(dataFolder=data_path)
    
    if str(device) != "cpu":
        if mode == "weak_supervision" and torch.cuda.device_count() != 1:
            gpu_count = torch.cuda.device_count() // 2
        else:
            gpu_count = torch.cuda.device_count()
        gpu_idx = torch.cuda.current_device()
        torch.cuda.set_device(gpu_idx)
    else:
        gpu_count = 0
        gpu_idx = 0


    # Only evaluation time
    if model is None:
        piece_imgs_path = total_imgs_path

        if backbone == "vgg":
            model = CRAFT()
        else:
            raise Exception("Undefined architecture")

        print("Loading weights from checkpoint (" + model_path + ")")
        if str(device) != "cpu":
            net_param = torch.load(model_path, map_location=f"cuda:{gpu_idx}")
            model.load_state_dict(copyStateDict(net_param["craft"]))
        else:
            net_param = torch.load(model_path, map_location="cpu")
            model.load_state_dict(copyStateDict(net_param["craft"]))
            model = model.cuda()
            cudnn.benchmark = False

    else:
        if gpu_count == 0:
            piece_imgs_path = total_imgs_path
        else:
            slice_idx = len(total_imgs_bboxes_gt) // gpu_count
            if gpu_idx == gpu_count - 1:
                piece_imgs_path = total_imgs_path[gpu_idx * slice_idx :]
            else:
                piece_imgs_path = total_imgs_path[gpu_idx * slice_idx : (gpu_idx + 1) * slice_idx]

    model.eval()

    # -----------------------------------------------------------------------------------------------------------------#
    total_imgs_bboxes_pre = []
    for k, img_path in enumerate(tqdm(piece_imgs_path)):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        single_img_bbox = []
        bboxes, polys, score_text = test_net(
            model,
            image,
            text_threshold = 0.75,
            link_threshold = 0.2,
            low_text = 0.5,
            device = device,
            poly = False
        )

        for box in bboxes:
            box_info = {"points": box, "text": "###", "ignore": False}
            single_img_bbox.append(box_info)
        total_imgs_bboxes_pre.append(single_img_bbox)

    results = []
    for i, (gt, pred) in enumerate(zip(total_imgs_bboxes_gt, total_imgs_bboxes_pre)):
        perSampleMetrics_dict = evaluator.evaluate_image(gt, pred)
        results.append(perSampleMetrics_dict)

    metrics = evaluator.combine_results(results)
    return metrics

def cal_eval(config, data, res_dir_name, opt, mode):
    evaluator = DetectionIoUEvaluator()
    test_config = DotDict(config.test[data])
    res_dir = os.path.join(os.path.join("exp", args.yaml), "{}".format(res_dir_name))

    if opt == "iou_eval":
        main_eval(
            config.test.trained_model,
            config.train.backbone,
            test_config,
            evaluator,
            res_dir,
            buffer=None,
            model=None,
            mode=mode,
        )
    else:
        print("Undefined evaluation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CRAFT Text Detection Eval")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="custom_data_train",
        type=str,
        help="Load configuration",
    )
    args = parser.parse_args()

    # load configure
    config = load_yaml(args.yaml)
    config = DotDict(config)

    val_result_dir_name = args.yaml
    cal_eval(
        config,
        "custom_data",
        val_result_dir_name + "-ic15-iou",
        opt="iou_eval",
        mode=None,
    )
