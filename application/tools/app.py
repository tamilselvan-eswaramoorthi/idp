# import flast module
from flask import Flask
from ocr import ocr_service
from parser import parser_service
# instance of flask application
app = Flask(__name__)

app.register_blueprint(ocr_service)
app.register_blueprint(parser_service)

@app.route("/heartbeat")
def hello_world():
    return "up"

if __name__ == '__main__':  
   app.run(host='0.0.0.0', port = 8001, debug = True)