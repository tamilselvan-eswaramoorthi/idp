# import flast module
from flask import Flask
from text_detector import detector_service
# instance of flask application
app = Flask(__name__)

app.register_blueprint(detector_service)

@app.route("/heartbeat")
def hello_world():
    return "up"

if __name__ == '__main__':  
   app.run(host='0.0.0.0', port = 8001, debug = True)