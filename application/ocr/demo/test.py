import cv2 
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from detection_model import CRAFT
from getboxes import get_textbox, get_image_list
from recognize_utils import get_text 
from recognize_model import Model 

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')


def get_detector(model_path, cudnn_benchmark=False):
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(model_path, map_location=device)))
    if device != 'cpu':
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark
    net.eval()
    return net

def get_recognizer(model_path, quantize=False, cudnn_benchmark=False):
    model = Model(num_class = 97)
    if device == 'cpu':
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        if quantize:
            try:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model

detector = get_detector("/home/tamilselvan/Desktop/idp/idp/application/tools/ocr/text_detector/weights/craft_mlt_25k.pth")
recognizer = get_recognizer('/home/tamilselvan/Desktop/idp/idp/application/tools/ocr/text_recognizer/weights/None-VGG-BiLSTM-CTC.pth')

image_path = 'sample.png'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
horizontal_list, free_list = get_textbox(detector, img, device)
image_list, max_width = get_image_list(horizontal_list, free_list, grey)
result = get_text(recognizer, max_width, image_list, device)

for res in result:
    print (res[-2])
