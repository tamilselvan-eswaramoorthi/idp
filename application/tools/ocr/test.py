from text_detector import Detector, loadImage
from text_recognizer import Recognizer

detect = Detector("text_detector/weights/craft_mlt_25k.pth")
recognizer = Recognizer('text_recognizer/weights')

image, grey = loadImage('sample.png')
horizontal_list, free_list = detect.get_textbox(image)
image_list, max_width = detect.get_image_list(horizontal_list, free_list, grey)
result = recognizer.get_text(max_width, image_list)


for res in result:
    print (res[-2])
