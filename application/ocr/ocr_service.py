import os
import fitz
import pandas as pd

import config
from ocr.text_detector import Detector
from ocr.text_recognizer import Recognizer

from ocr.utils import image_ops, convert

class Processor:
    def __init__(self) -> None:
        self.detect = Detector(os.environ.get("DETECTOR_PATH"))
        self.recognize = Recognizer(os.environ.get("RECOGNIZER_PATH"))
    
    def _process_image(self, path: str):
        bboxes, words = [], []
        image, gray = image_ops.loadImage(path)
        horizontal_list, free_list, _ = self.detect.get_textbox(image)
        image_list, max_width = self.detect.get_image_list(horizontal_list, free_list, gray)
        bboxes, words = self.recognize.get_text(max_width, image_list, is_one_list = True)
        return {
            "image_path": path,
            "bboxes": bboxes,
            "words": words
            }
    
    def _process_pdf(self, path: str):
        doc = fitz.open(path)
        file_name = os.path.basename(path)   

        results = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            words = page.get_text("words")  

            word_list = []
            bbox_list = []

            for w in words:
                bbox = w[:4]  # bbox (x0, y0, x1, y1)
                word = w[4]  # word text
                bbox_list.append(bbox)
                word_list.append(word)

            results.append({
                "words": word_list, 
                "bboxes": bbox_list, 
                "page_number": page_num,
                "file_name": file_name
                })
        return results
    
    def _process_excel(self, path: str):
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            df = pd.read_excel(path)
        else:
            return {"status": 500, "message":"Unsupported file format"}
        
        pdf_path = os.path.splitext(path)[0] + '.pdf'
        convert.dataframe_to_pdf(df, pdf_path)
        return self._process_pdf(pdf_path)

        
    def process_files(self, files: list): 
        results = []
        for file in files:
            path = os.path.join(config.base_dir, file.filename)
            file.save(path)
            _, extension = os.path.splitext(os.path.basename(path))
            if extension in config.image_extensions:
                results.append({"type": "image", "ocr": self._process_image(path)})
            elif extension in config.excel_extensions:
                results.append({"type": "excel", "ocr": self._process_excel(path)})
            elif extension in config.pdf_extension:
                results.append({"type": "pdf", "ocr": self._process_pdf(path)})
            elif extension in config.docx_extension:
                pass #TODO 
            else:
                return {"status": 500, "message": f"file -- {path} not supported"}
                
            return {"status": 200, "message": "success", "results": results}

