import cv2
import numpy as np
from skimage import util
import base64
import tensorflow as tf
import tensorflow_addons as tfa
import time
from flask import Flask, request
import json

from Flaw_detection_FEEC import *

# limitando o uso de RAM da GPU
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

models_path = 'projeto/ViT_models/'
m1,m2,m3,m4 = loading_models(models_path)

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>ViT model</h1>"

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        start = time.time()
        img_json = request.get_json()
        img_json = json.loads(img_json)
        stop = time.time()

        time_request = stop - start
        encoded_data = img_json

        ################ Decoding the images ##################
        start = time.time()
        d_data = decode_dic(encoded_data)
        stop = time.time()
        time_request = stop - start
        #######################################################
        
        ############### Remove the bussbars ###################
        start = time.time()
        res=processing_images(d_data)
        stop = time.time()
        time_preprocessing = stop - start
        #######################################################
        
        ############## Do the predictions ####################
        start = time.time()
        # preds, out_imgs, aff_ar = ml_predict_bs(d_data, res, m1, m2, m3, m4)
        dic_out = ml_predict_bs(d_data,res, m1, m2, m3, m4)
        stop = time.time()
        time_processing = stop - start
        ######################################################
        
        ########### Enconde the results ########################
        start = time.time()
        #enc_res=encode_response(encoded_data, preds, aff_ar, out_imgs)
        enc_res = encode_response(dic_out)
        stop = time.time()
        time_postprocessing = stop - start
        ########################################################

        enc_res['request_time'] = str(time_request)
        enc_res['preprocessing_time'] = str(time_preprocessing)
        enc_res['processing_time'] = str(time_processing)
        enc_res['postprocessing_time'] = str(time_postprocessing)

        response = json.dumps(enc_res)
        return response

if __name__=="__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=7000)
