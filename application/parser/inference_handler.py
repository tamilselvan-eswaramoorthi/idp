
import os
import json
import torch
import logging
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, AutoProcessor, LayoutLMv3Processor

from .utils import image_label_2_color, normalize_box, compare_boxes, adjacent

logger = logging.getLogger(__name__)

def get_flattened_output(docs):
    flattened_output = []
    annotation_key = 'output'
    for doc in docs:
        flattened_output_item = {annotation_key: []}
        doc_annotation = doc[annotation_key]
        for i, span in enumerate(doc_annotation):
            if len(span['words']) > 1:
                for span_chunk in span['words']:
                    flattened_output_item[annotation_key].append({
                        'label': span['label'],
                        'text': span_chunk['text'],
                        'words': [span_chunk]
                    })
            else:
                flattened_output_item[annotation_key].append(span)
        flattened_output.append(flattened_output_item)
    return flattened_output

def annotate_image(image_path, annotation_object):
    img = None
    image = Image.open(image_path).convert('RGBA')
    tmp = image.copy()
    label2color = image_label_2_color(annotation_object)
    overlay = Image.new('RGBA', tmp.size, (0, 0, 0)+(0,))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    predictions = [span['label'] for span in annotation_object['output']]
    boxes = [span['words'][0]['box'] for span in annotation_object['output']]
    for prediction, box in zip(predictions, boxes):
        draw.rectangle(box, outline=label2color[prediction],
                        width=3, fill=label2color[prediction]+(int(255*0.33),))
        draw.text((box[0] + 10, box[1] - 10), text=prediction,
                fill=label2color[prediction], font=font)

    img = Image.alpha_composite(tmp, overlay)
    img = img.convert("RGB")

    image_name = os.path.basename(image_path)
    image_name = image_name[:image_name.find('.')]
    img.save(f'/content/{image_name}_inference.jpg')

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.model = None
        self.model_dir = None
        self.device = 'cpu'
        self.error = None
        self.initialized = False
        self._raw_input_data = None
        self._processed_data = None
        self._images_size = None

    def initialize(self, model_dir):
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        # self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        # assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        inference_dict = batch
        self._raw_input_data = inference_dict
        images = [Image.open(path).convert("RGB") for path in inference_dict['image_path']]
        self._images_size = [img.size for img in images]
        words = inference_dict['words']
        boxes = [[normalize_box(box, images[i].size[0], images[i].size[1])for box in doc] for i, doc in enumerate(inference_dict['bboxes'])]
        encoded_inputs = self.processor(images, words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        self._processed_data = encoded_inputs
        return encoded_inputs

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # TODO load the model state_dict before running the inference
        # Do some inference call to engine here and return output
        with torch.no_grad():
            inference_outputs = self.model(**model_input)
            predictions = inference_outputs.logits.argmax(-1).tolist()
        results = []
        for i in range(len(predictions)):
            tmp = dict()
            tmp[f'output_{i}'] = predictions[i]
            results.append(tmp)

        return [results]

    def postprocess(self, inference_output):
        docs = []
        k = 0
        for page, doc_words in enumerate(self._raw_input_data['words']):
            doc_list = []
            width, height = self._images_size[page]
            for i, doc_word in enumerate(doc_words, start=0):
                word_tagging = None
                word_labels = []
                word = dict()
                word['id'] = k
                k += 1
                word['text'] = doc_word
                word['pageNum'] = page + 1
                word['box'] = self._raw_input_data['bboxes'][page][i]
                _normalized_box = normalize_box(
                    self._raw_input_data['bboxes'][page][i], width, height)
                for j, box in enumerate(self._processed_data['bbox'].tolist()[page]):
                    if compare_boxes(box, _normalized_box):
                        if self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]] != 'O':
                            word_labels.append(
                                self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]][2:])
                        else:
                            word_labels.append('other')
                if word_labels != []:
                    word_tagging = word_labels[0] if word_labels[0] != 'other' else word_labels[-1]
                else:
                    word_tagging = 'other'
                word['label'] = word_tagging
                word['pageSize'] = {'width': width, 'height': height}
                if word['label'] != 'other':
                    doc_list.append(word)
            spans = []
            def adjacents(entity): return [
                adj for adj in doc_list if adjacent(entity, adj)]
            output_test_tmp = doc_list[:]
            for entity in doc_list:
                if adjacents(entity) == []:
                    spans.append([entity])
                    output_test_tmp.remove(entity)

            while output_test_tmp != []:
                span = [output_test_tmp[0]]
                output_test_tmp = output_test_tmp[1:]
                while output_test_tmp != [] and adjacent(span[-1], output_test_tmp[0]):
                    span.append(output_test_tmp[0])
                    output_test_tmp.remove(output_test_tmp[0])
                spans.append(span)

            output_spans = []
            for span in spans:
                if len(span) == 1:
                    output_span = {"text": span[0]['text'],
                                   "label": span[0]['label'],
                                   "words": [{
                                       'id': span[0]['id'],
                                       'box': span[0]['box'],
                                       'text': span[0]['text']
                                   }],
                                   }
                else:
                    output_span = {"text": ' '.join([entity['text'] for entity in span]),
                                   "label": span[0]['label'],
                                   "words": [{
                                       'id': entity['id'],
                                       'box': entity['box'],
                                       'text': entity['text']
                                   } for entity in span]

                                   }
                output_spans.append(output_span)
            docs.append({f'output': output_spans})
        return [json.dumps(docs, ensure_ascii=False)]

    def handle(self, data):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        inference_out = self.postprocess(model_out)[0]
        with open('LayoutlMV3InferenceOutput.json', 'w') as inf_out:
            inf_out.write(inference_out)
        inference_out_list = json.loads(inference_out)
        flattened_output_list = get_flattened_output(inference_out_list)
        for i, flattened_output in enumerate(flattened_output_list):
            annotate_image(data['image_path'][i], flattened_output)
