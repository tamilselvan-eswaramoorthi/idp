import cv2
import json
import requests
from uuid import uuid4
from flask import Blueprint, request

from .layoutlm import Parser
    
parser_service = Blueprint('parser', __name__, url_prefix = '/parser')

parser = Parser()

@parser_service.route("/get_results", methods = ["POST"])
def get_results():
    input_data = json.dumps(request.json)

    response = requests.post("http://localhost:8001/ocr/get_results", data=input_data, headers = {'Content-Type': 'application/json'})
    if response.status_code == 200:
        response = response.json()
        if response['status'] == 200:
            result = parser.process_image(response['result'])
    result = result.save("geeks.jpg")
    return ""
