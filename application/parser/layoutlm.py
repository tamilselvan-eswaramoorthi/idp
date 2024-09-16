import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification


class Parser:

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice")
        labels = ['O', 'B-ABN', 'B-BILLER', 'B-BILLER_ADDRESS', 'B-BILLER_POST_CODE', 'B-DUE_DATE', 'B-GST', 'B-INVOICE_DATE', 'B-INVOICE_NUMBER', 'B-SUBTOTAL', 'B-TOTAL', 'I-BILLER_ADDRESS']
        self.id2label = {v: k for v, k in enumerate(labels)}
        self.label2color = {label: "red" for label in labels}

    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    def draw_boxes_on_img(self, preds_or_labels, boxes, draw, width, height, unnormalize = False):        
        for pred_or_label, box in zip(preds_or_labels, boxes):
            label = self.id2label(pred_or_label).lower()

            if label == 'other':
                continue
            else:
                if unnormalize:
                    box = self.unnormalize_box(box, width, height)
                
                color = self.label2color[label]
                draw.rectangle(box, outline=color)
                draw.text((box[0] + 10, box[1] - 10), text=label, fill=color)


    def process_image(self, ocr_results):
        for ocr_result in ocr_results:
            image = Image.open(ocr_result['image_path'])
            width, height = image.size
            bboxes = ocr_result['bboxes']

            # bboxes = [[max(0, min(x, 1000)), max(0, min(y, 1000)), max(0, min(z, 1000)), max(0, min(w, 1000))] for [x, y, z, w] in ocr_result['bboxes']]


            encoding = self.processor(image, 
                                      text=ocr_result['words'], 
                                      boxes=bboxes, 
                                      return_offsets_mapping = True,
                                      return_tensors='pt')
            offset_mapping = encoding.pop('offset_mapping')

            outputs = self.model(**encoding)

            # get predictions
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = encoding.bbox.squeeze().tolist()

            # only keep non-subword predictions
            is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
            true_predictions = [self.id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
            true_boxes = [self.unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
            print (token_boxes, true_boxes)
            # draw predictions over the image
            draw = ImageDraw.Draw(image)
            for prediction, box in zip(true_predictions, token_boxes):
                draw.rectangle(box, outline=self.label2color[prediction])
                draw.text((box[0]+10, box[1]-10), text=prediction, fill=self.label2color[prediction])
        
        return image