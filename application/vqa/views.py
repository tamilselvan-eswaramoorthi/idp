import os
import re
import cv2
import torch
from PIL import Image
from flask import Blueprint, request
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering

import config 
from ocr import ocr_process

class LayoutLMV3Handler:
    def __init__(self, model_name) -> None:
        # Initialize LayoutLMv3 processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

        # self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
        self.model =  AutoModelForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-sroie")


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def normalize_bbox(self, bbox, width=1000, height=1000):
        x0, y0, x1, y1 = bbox
        return [
            int((x0 / width) * 1000),
            int((y0 / height) * 1000),
            int((x1 / width) * 1000),
            int((y1 / height) * 1000),
        ]
    
    def process_vqa(self, file_name):
        results = ocr_process.process_files(file_name) 
        if results['status'] == 200:
            normalized_bboxes = [self.normalize_bbox(bbox) for bbox in results['result'][0]["ocr"]["bboxes"]]
            encoded_inputs = self.processor(
                text= results['result'][0]["ocr"]["words"],
                boxes=normalized_bboxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512  # Adjust based on your text length
            )
            outputs = self.model(**encoded_inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            return predictions


    # def process_vqa(self, input_img, question):
    #     # Convert the input image to a PIL Image
    #     image = Image.fromarray(input_img).convert("RGB")

    #     # Generate encoding using LayoutLMv3 processor
    #     encoding = self.processor(image, question, return_tensors="pt")

    #     # Extract pixel values, bounding boxes, and input IDs
    #     input_ids = encoding.input_ids.to(self.device)
    #     attention_mask = encoding.attention_mask.to(self.device)
    #     bbox = encoding.bbox.to(self.device)
    #     pixel_values = encoding.pixel_values.to(self.device)

    #     # Perform forward pass for question answering
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         bbox=bbox,
    #         pixel_values=pixel_values
    #     )

    #     # Extract start and end logits
    #     start_logits = outputs.start_logits
    #     end_logits = outputs.end_logits

    #     # Get the most probable start and end positions
    #     start_index = torch.argmax(start_logits, dim=1).item()
    #     end_index = torch.argmax(end_logits, dim=1).item() + 1

    #     # Decode the answer from input IDs
    #     answer = self.processor.tokenizer.decode(input_ids[0][start_index:end_index])
    #     return answer


func = LayoutLMV3Handler("microsoft/layoutlmv3-base")

donut_service = Blueprint('vqa', __name__, url_prefix = '/vqa')

@donut_service.route("/vqa", methods = ["POST"])
def vqa():
    files = request.files.getlist('path')
    question = request.form.get('question')

    if not len(files):
        return {"status": 500, "message": "path not specified"}
    
    if isinstance(files, str):
        files = [files] 

    results = []
    for file in files:
        path = os.path.join(config.base_dir, file.filename)
        file.save(path)
        image = cv2.imread(path)
        response = func.process_vqa(path, question=question)
        results.append(response)
    
    return {
        "status": 200,
        "message": "success",
        "results": results
    }
