import os
import cv2
import json
import yaml
import time
import argparse
import torch
import numpy as np
import gc

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
from struct_eqtable import build_model

from extract_pdf.utils import load_pdf_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor

def crop_img(input_res, input_pil_img, padding_x=0, padding_y=0):
    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])
    # Create a white background with an additional width and height of 50
    crop_new_width = crop_xmax - crop_xmin + padding_x * 2
    crop_new_height = crop_ymax - crop_ymin + padding_y * 2
    return_image = Image.new('RGB', (crop_new_width, crop_new_height), 'white')

    # Crop image
    crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    cropped_img = input_pil_img.crop(crop_box)
    return_image.paste(cropped_img, (padding_x, padding_y))
    return_list = [padding_x, padding_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width, crop_new_height]
    return return_image, return_list

def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor

def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model

def tr_model_init(weight, max_time, device='cuda'):
    tr_model = build_model(weight, max_new_tokens=4096, max_time=max_time)
    if device == 'cuda':
        tr_model = tr_model.cuda()
    return tr_model

class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    ## ======== model init ========##
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    img_size = model_configs['model_args']['img_size']
    conf_thres = model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device = model_configs['model_args']['device']
    dpi = model_configs['model_args']['pdf_dpi']
    mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
    mfr_transform = transforms.Compose([mfr_vis_processors, ])
    tr_model = tr_model_init(model_configs['model_args']['tr_weight'], max_time=model_configs['model_args']['table_max_time'], device=device)
    layout_model = layout_model_init(model_configs['model_args']['layout_weight'])

    ## ======== model init ========##
    
    if os.path.isdir(args.pdf):
        all_pdfs = [os.path.join(args.pdf, name) for name in os.listdir(args.pdf)]
    else:
        all_pdfs = [args.pdf]

    print("total files:", len(all_pdfs))

    for pdf_idx, single_pdf in enumerate(all_pdfs):
        try:
            img_list = load_pdf_fitz(single_pdf, dpi=dpi)
        except:
            img_list = None
            print("unexpected pdf file:", single_pdf)
        if img_list is None:
            continue
        print("pdf index:", pdf_idx, "pages:", len(img_list))

        doc_layout_result = []
        for idx, image in enumerate(img_list):
            img_H, img_W = image.shape[0], image.shape[1]
            layout_res = layout_model(image, ignore_catids=[])
            doc_layout_result.append(layout_res)

        # ocr and table recognition
        for idx, image in enumerate(img_list):

            layout_res = doc_layout_result[idx]['layout_dets']
            pil_img = Image.fromarray(image)

            ocr_res_list = []
            table_res_list = []
            single_page_mfdetrec_res = []

            for res in layout_res:
                if int(res['category_id']) in [13, 14]:
                    single_page_mfdetrec_res.append({
                        "bbox": [int(res['poly'][0]), int(res['poly'][1]),
                                 int(res['poly'][4]), int(res['poly'][5])],
                    })
                elif int(res['category_id']) in [0, 1, 2, 4, 6, 7]:
                    ocr_res_list.append(res)
                elif int(res['category_id']) in [5]:
                    table_res_list.append(res)

            ocr_start = time.time()
            # Process each area that requires OCR processing
            for res in ocr_res_list:
                new_image, useful_list = crop_img(res, pil_img, padding_x=50, padding_y=50)
                paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
                # Adjust the coordinates of the formula area
                adjusted_mfdetrec_res = []
                for mf_res in single_page_mfdetrec_res:
                    mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
                    # Adjust the coordinates of the formula area to the coordinates relative to the cropping area
                    x0 = mf_xmin - xmin + paste_x
                    y0 = mf_ymin - ymin + paste_y
                    x1 = mf_xmax - xmin + paste_x
                    y1 = mf_ymax - ymin + paste_y
                    # Filter formula blocks outside the graph
                    if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
                        continue
                    else:
                        adjusted_mfdetrec_res.append({
                            "bbox": [x0, y0, x1, y1],
                        })

                # OCR recognition
                new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)
                ocr_res = ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res)[0]

                # Integration results
                if ocr_res:
                    for box_ocr_res in ocr_res:
                        p1, p2, p3, p4 = box_ocr_res[0]
                        text, score = box_ocr_res[1]

                        # Convert the coordinates back to the original coordinate system
                        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
                        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
                        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
                        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]

                        layout_res.append({
                            'category_id': 15,
                            'poly': p1 + p2 + p3 + p4,
                            'score': round(score, 2),
                            'text': text,
                        })

            ocr_cost = round(time.time() - ocr_start, 2)
            print(f"ocr cost: {ocr_cost}")

            table_start = time.time()
            for res in table_res_list:
                new_image, _ = crop_img(res, pil_img)

                single_table_start = time.time()
                with torch.no_grad():
                    output = tr_model(new_image)
                if (time.time() - single_table_start) > model_configs['model_args']['table_max_time']:
                    res["timeout"] = True
                res["latex"] = output[0]
            table_cost = round(time.time() - table_start, 2)
            print(f"table cost: {table_cost}")

        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(single_pdf)[0:-4]
        with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
            json.dump(doc_layout_result, f)
        