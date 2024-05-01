import os
import torch
import shutil
from flask import Blueprint, request

from ocr.text_detector import Detector
from ocr.text_recognizer import Recognizer
from ocr.utils.load_image import loadImage

# from ocr.text_detector import Trainer
# from ocr.text_detector.trainer.config.load_config import load_yaml, DotDict


ocr_service = Blueprint('ocr', __name__, url_prefix = '/ocr')

# if os.environ.get("MODE") == "prod":
detect = Detector("/home/tamilselvan/Desktop/idp/idp/application/tools/ocr/text_detector/weights/craft_mlt_25k.pth")
recognize = Recognizer("/home/tamilselvan/Desktop/idp/idp/application/tools/ocr/text_recognizer/weights/")

@ocr_service.route("/train_detector", methods = ["POST"])
def train_detector():
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

@ocr_service.route("/get_results", methods = ["POST"])
def get_results():
    input_data = request.json

    images_path = input_data.get("images_path", None)

    if images_path is None:
        return {"status": 500, "message": "Image path not specified"}
    
    if isinstance(images_path, str):
        images_path = [images_path]

    result_dict = {"status": 200, "message": "success"}

    results = []
    for image_path in images_path:
        image, gray = loadImage(image_path)
        horizontal_list, free_list, _ = detect.get_textbox(image)
        image_list, max_width = detect.get_image_list(horizontal_list, free_list, gray)
        boxes, words = recognize.get_text(max_width, image_list, is_one_list = True)
        results.append({
        "image_path": image_path,
        "bboxes": boxes,
        "words": words
    })

    result_dict['result'] = results
    return result_dict