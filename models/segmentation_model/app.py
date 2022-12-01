from flask import Flask, request
import json
from json import JSONEncoder
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import base64
import time
import yaml

def criterio(pred,img_mp) -> list:
    area=[0,0,0] # area comprometida em cada anomalia [trinca, solda , outros]  
    if np.sum(pred)>0:  # if true é porque tem anomalia 
        if pred[0]>0: # a celula tem trinca
            area[0]=np.sum(np.abs(img_mp[:,:,1]))/(300*150)*100 # sai em porcentagem.
        if pred[1]>0: # a celula tem solda
            area[1]=np.sum(np.abs(img_mp[:,:,2]))/(300*150)*100 # sai em porcentagem.  
        if pred[2]>0: # a celula tem outos
            area[2]=np.sum(np.abs(img_mp[:,:,1]))/(300*150)*100+np.sum(np.abs(img_mp[:,:,2]))/(300*150)*100 # sai em porcentagem.
    return area

path_1 = "projeto/models/ucnn.h5"
path_2 = "projeto/models/unet.h5"


m1 = load_model(path_1)
m2 = load_model(path_2)

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Semantic segmentation model</h1>"

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        start = time.time()

        # loading config data
        with open('projeto/data/config.yml') as file:
            config = yaml.safe_load(file)

        batch_size = config["models"]["segmentation_model"]["batch_size"]
        use_cuda = config["models"]["segmentation_model"]["use_cuda"]
        memory_limit = config["models"]["segmentation_model"]["memory_limit"]

        # limitando o uso de RAM da GPU
        if use_cuda:
            gpus = tf.config.list_physical_devices("GPU")
            print(gpus)
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

        img_json = request.get_json()
        img_json = json.loads(img_json)
        stop = time.time()
        time_request = stop - start

        start = time.time()
        keys = list(img_json.keys())
        x = []
        for k in keys:
            im = np.frombuffer(base64.b64decode(img_json[k]), dtype=np.uint8).reshape(300,150)
            im = np.asarray(im, dtype=np.float32)
            im = (im - np.min(im)) / (np.max((im - np.min(im))))
            x.append(im)
        del img_json
        x = np.array(x, dtype=np.float32)
        x = np.expand_dims(x, -1) # (num_figs,300,150,1)
        stop = time.time()
        time_preprocessing = stop - start

        start = time.time()
        output_data_1 = []
        output_data_2 = []
        if use_cuda:
            dev = "gpu"
        else:
            dev = "cpu"
        with tf.device(dev):
            for k in range(0,x.shape[0], batch_size):
                try:
                    xi = x[k:k+batch_size] # (batch_size,300,150,1)
                except:
                    xi = x[k:]
                pred_1 = m1.predict(xi)
                pred_1[pred_1>1] = 1
                pred_2 = m2.predict(pred_1)
                pred_2[pred_2>=0.8] = 1
                pred_2[pred_2!=1] = 0
                #pred_2 = np.around(pred_2)

                output_data_1.append(pred_1)
                output_data_2.append(pred_2)
        output_data_1 = np.vstack(output_data_1)
        output_data_2 = np.vstack(output_data_2)  
        stop = time.time()
        time_processing = stop - start

        out = {}
        cont = 0
        start = time.time()
        for k in keys:
            im = x[[cont]]

            img = output_data_1[cont].squeeze() # (300,150,3), [0,1]
            img255 = (img - img.min())/(img.max() - img.min())
            img255 = np.array(255*img255, dtype=np.uint8)
            img255 = base64.b64encode(img255).decode('utf-8')
            
            pred = f'{int(output_data_2[cont][0])}{int(output_data_2[cont][1])}{int(output_data_2[cont][2])}'
            i = 0
            pred = '000'  # tr-sf-ot
            tam = criterio(output_data_2[cont], img)

            if int(output_data_2[cont][0]) == 1:
                pred = '100'
                out[k+f"_tamanho_{i}"] = tam[0]
                out[k+f"_label_{i}"] = pred
                i+=1
            if int(output_data_2[cont][1]) == 1:
                pred = '010'
                out[k+f"_tamanho_{i}"] = tam[1]
                out[k+f"_label_{i}"] = pred
                i+=1  
            if int(output_data_2[cont][2]) == 1:
                pred = '001'
                out[k+f"_tamanho_{i}"] = tam[2]
                out[k+f"_label_{i}"] = pred
                i+=1         
            if pred != '000':
                out[k] = img255
            cont += 1
        stop = time.time()
        time_postprocessing = stop - start
        
        out['request_time'] = str(time_request)
        out['preprocessing_time'] = str(time_preprocessing)
        out['processing_time'] = str(time_processing)
        out['postprocessing_time'] = str(time_postprocessing)

        response = json.dumps(out)
        return response

if __name__=="__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=4000)