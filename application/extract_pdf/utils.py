import re
import fitz
import numpy as np
from PIL import Image

def load_pdf_fitz(pdf_path, dpi=72):
    images = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pm = page.get_pixmap(matrix=mat, alpha=False)

        # If the width or height exceeds 9000 after scaling, do not scale further.
        if pm.width > 9000 or pm.height > 9000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        images.append(np.array(img))
    return images


def layout_rm_equation(layout_res):
    rm_idxs = []
    for idx, ele in enumerate(layout_res['layout_dets']):
        if ele['category_id'] == 10:
            rm_idxs.append(idx)
    
    for idx in rm_idxs[::-1]:
        del layout_res['layout_dets'][idx]
    return layout_res


def get_croped_image(image_pil, bbox):
    x_min, y_min, x_max, y_max = bbox
    croped_img = image_pil.crop((x_min, y_min, x_max, y_max))
    return croped_img


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code.
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s