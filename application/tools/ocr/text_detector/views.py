import os
import cv2
import shutil
from flask import Blueprint, request

from text_detector import Detector

from text_detector.trainer.train import Trainer
from text_detector.utils.imgproc import loadImage
from text_detector.trainer.config.load_config import load_yaml, DotDict


detector_service = Blueprint('detector', __name__, url_prefix = '/text_detector')

yaml_path = 'text_detector/trainer/config/custom_data_train'

@detector_service.route("/train", methods = ["POST"])
def train():
    config = dict()
    ## todo
    config = load_yaml(yaml_path)
    print (config)
    # Make result_dir
    res_dir = os.path.join(config["results_dir"], yaml_path)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy(yaml_path + ".yaml", os.path.join(res_dir, os.path.basename(yaml_path)) + ".yaml")

    if config["mode"] == "weak_supervision":
        mode = "weak_supervision"
    else:
        mode = None


    trainer = Trainer(DotDict(config), 0, mode)
    trainer.train({"custom_data":None})

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