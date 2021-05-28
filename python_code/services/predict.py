from flask import Flask
from flask_restful import Api, Resource
from flask_restful import Resource, reqparse
import werkzeug
from flask import render_template
import io
from base64 import encodebytes
from PIL import Image
from flask import jsonify

import numpy as np
import sys


from extract_car_num import CarNumberDetector



app = Flask(__name__)
api = Api(app)

yolo_model_path="../yolo/viden_trained_models/viden_yolo.h5"
crnn_model_path="../yolo/viden_trained_models/viden_crnn.h5"
classes_path="../yolo/classes.txt"
output_path="../output/"

@app.route("/")
def index():
    return render_template("index.html")

class ProcessImageEndpoint(Resource):
    def __init__(self):
        # Create a request parser
        parser = reqparse.RequestParser()
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        # Sending more info in the form? simply add additional arguments, with the location being 'form'
        # parser.add_argument("other_arg", type=str, location='form')
        self.req_parser = parser
        self.cd = CarNumberDetector(yolo_model_path,crnn_model_path,classes_path,output_path)
        # This method is called when we send a POST request to this endpoint

    def post(self):
        # The image is retrieved as a file
        image_file = self.req_parser.parse_args(strict=True).get("image", None)

        if image_file:
            # Get the byte content using `.read()`
            image = image_file.read()
            image = Image.open(io.BytesIO(image))
            img_arr=np.asarray(image)
            image,car_num=self.cd.extract_number(image_array=img_arr,save=True)
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG') # convert the PIL image to byte array
            encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
            # Now do something with the image...
            return jsonify( licenseNumber=car_num,carImage=encoded_img )
		
        else:
            return "No image sent :("
		
		
api.add_resource(ProcessImageEndpoint, '/predict')
app.run(debug=True)
