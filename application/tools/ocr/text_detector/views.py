import os
import cv2
import torch
import shutil
from flask import Blueprint, request

from text_detector import Detector

from text_detector.trainer.train import Trainer
from text_detector.utils.imgproc import loadImage
from text_detector.trainer.config.load_config import load_yaml, DotDict


detector_service = Blueprint('detector', __name__, url_prefix = '/text_detector')


@detector_service.route("/train", methods = ["POST"])
def train():
    yaml_path = request.form.get("yaml_path", 'text_detector/trainer/config/custom_data_train')
    config = load_yaml(yaml_path)
    config["results_dir"] = res_dir = request.form.get("results_dir", "./work_dir")
    config["data_root_dir"] = request.form.get("data_root_dir", "./data_root_dir/")
    config["train"]["ckpt_path"] = request.form.get("ckpt_path", "text_detector/weights/CRAFT_clr_amp_29500.pth")
    config["train"]["eval_interval"] = int(request.form.get("eval_interval", 1000))
    config["train"]["batch_size"] = int(request.form.get("batch_size", 8))
    config["train"]["st_iter"] = int(request.form.get("st_iter", 0))
    config["train"]["end_iter"] = int(request.form.get("end_iter", 25000))
    mode = request.form.get("mode", "weak_supervision")

    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    shutil.copy(yaml_path + ".yaml", 
                os.path.join(res_dir, os.path.basename(yaml_path)) + ".yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Trainer(DotDict(config), device = device, mode = mode).train()

@detector_service.route("/test", methods = ["POST"])
def test():
    detect = Detector("text_detector/weights/craft_mlt_25k.pth")
    
    image_path = request.form.get("image_path", None)
    if image_path is None:
        return {"status": 500, "message": "Image path not specified"}
    
    image = loadImage(image_path)
    horizontal_list, free_list, result_image = detect.get_textbox(image)
    result_path = 'temp.png'
    cv2.imwrite(result_path, result_image)
    return {
        "status": 200,
        "message": "success", 
        "result": {
            "res_path": result_path
            }
        }