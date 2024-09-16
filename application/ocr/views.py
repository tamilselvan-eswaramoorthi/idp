import os
import torch
import shutil
from flask import Blueprint, request

from ocr.ocr_service import Processor

from ocr.text_detector import Trainer
from ocr.text_detector.trainer.config.load_config import load_yaml, DotDict


ocr_service = Blueprint('ocr', __name__, url_prefix = '/ocr')

ocr_process = Processor()

@ocr_service.route("/train_detector", methods = ["POST"])
def train_detector():
    yaml_path = request.form.get("yaml_path", 'ocr/text_detector/trainer/config/custom_data_train')
    config = load_yaml(yaml_path)
    config["results_dir"] = res_dir = request.form.get("results_dir", "./work_dir")
    config["data_root_dir"] = request.form.get("data_root_dir", "./data_root_dir/")
    config["train"]["ckpt_path"] = request.form.get("ckpt_path", "ocr/text_detector/weights/CRAFT_clr_amp_29500.pth")
    config["train"]["eval_interval"] = int(request.form.get("eval_interval", 1000))
    config["train"]["batch_size"] = int(request.form.get("batch_size", 8))
    config["train"]["st_iter"] = int(request.form.get("st_iter", 0))
    config["train"]["end_iter"] = int(request.form.get("end_iter", 25000))
    mode = request.form.get("mode", "weak_supervision")
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    shutil.copy(yaml_path + ".yaml", os.path.join(res_dir, os.path.basename(yaml_path)) + ".yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Trainer(DotDict(config), device = device, mode = mode).train()

@ocr_service.route("/get_results", methods = ["POST"])
def get_results():
    files = request.files.getlist('path')
    if not len(files):
        return {"status": 500, "message": "path not specified"}
    
    if isinstance(files, str):
        files = [files] 

    results = ocr_process.process_files(files) 
    if results['status'] != 200: 
        return results 
    
    return {
        "status": 200, 
        "message": "success", 
        "result": results['results']
        } 
    