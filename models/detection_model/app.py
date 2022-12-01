from flask import Flask, request
import json
from json import JSONEncoder
import torch
import numpy as np
import torchvision
import cv2
import base64
import time
import yaml

model = torch.load("./projeto/models/modelo_rcnn.ckpt", map_location=torch.device("cpu"))
model.eval()

# aplicativo flask
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Object detection model</h1>"

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        start = time.time()
        
        # loading config data
        with open('./projeto/data/config.yml') as file:
            config = yaml.safe_load(file)

        batch_size = config["models"]["detection_model"]["batch_size"]
        use_cuda = config["models"]["detection_model"]["use_cuda"]
        
        if use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        
        img_json = request.get_json()
        img_json = json.loads(img_json)
        stop = time.time()
        time_request = stop - start

        start = time.time()
        keys = list(img_json.keys())
        x = []
        for k in keys:
            img = np.frombuffer(base64.b64decode(img_json[k]), dtype=np.uint8).reshape(300,150)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.expand_dims(img, -1) # 150x300x1
            img = np.tile(img, (1,1,3)) # 150x300x3
            x.append(img)
        x = np.array(x)
        x = torch.Tensor(x)
        x = torch.permute(x, (0,3,1,2))
        stop = time.time()
        time_preprocessing = stop - start

        start = time.time()
        with torch.inference_mode():
            total_size = x.shape[0]
            pred_model = []
            for k in range(0, total_size, batch_size):
                idx = np.arange(k, k+batch_size)
                idx = idx[idx < total_size]
                x_temp = x[idx].to(device)
                pred_model += model(x_temp) # o modelo retorna uma lista onde cada entrada é um elemento do batch
                del(x_temp)
                torch.cuda.empty_cache()
            del x

            stop = time.time()
            time_processing = stop - start

            model.to("cpu") # volta para a cpu
            out_dict = {}
            
            start = time.time()
            for k in range(len(pred_model)): # criando imagens para cada predicao do batch
                out = pred_model[k]

                out['labels'][out['labels']==1] = 1
                out['labels'][out['labels']==2] = 2
                out['labels'][out['labels']==3] = 3
                out['labels'][out['labels']==4] = 3
                out['labels'][out['labels']==5] = 3

                nms = torchvision.ops.batched_nms(boxes = out['boxes'],
                                                scores = out['scores'],
                                                idxs = out['labels'],
                                                iou_threshold = 0.1)
                out_nms = {
                    'boxes': [],
                    'labels': [],
                    'scores': []
                    }

                for n in range(len(out['labels'])):
                    if n in nms:
                        out_nms['boxes'].append(out['boxes'][n].to("cpu").numpy())
                        out_nms['labels'].append(out['labels'][n].to("cpu").numpy())
                        out_nms['scores'].append(out['scores'][n].to("cpu").numpy())

                if len(out_nms['labels'])>0:
                    out_nms['boxes'] = np.vstack(out_nms['boxes'])
                    out_nms['labels'] = np.hstack(out_nms['labels'])
                    out_nms['scores'] = np.hstack(out_nms['scores'])	

                im = np.frombuffer(base64.b64decode(img_json[keys[k]]), dtype=np.uint8).reshape(300,150)
                im = (im - im.min())/(im.max() - im.min())
                im = np.expand_dims(im, -1)
                im = np.tile(im, (1,1,3))
                im = np.array(255*im, dtype=np.uint8)
                
                falha = 0 # flag utilizado para indicar se ha pelo menos uma falha na celula k
                cont = 0
                for n in range(len(out_nms["labels"])): # se tiver mais de uma falha na imagem
                    if out_nms["scores"][n] > 0.3: # limite do score para considerar uma falha
                        x0 = int(out_nms["boxes"][n, 0])
                        y0 = int(out_nms["boxes"][n, 1])
                        x1 = int(out_nms["boxes"][n, 2])
                        y1 = int(out_nms["boxes"][n, 3])
                        col = (255, 0, 0)
                        trinca = 0
                        sf = 0
                        ot = 0
                        if out_nms["labels"][n] in [2]:
                            falha = 1 # muda o flag
                            trinca = 1
                            col = (0,255,0)
                            tam = 100*(np.abs(x1-x0)*np.abs(y1-y0))/(300*150)

                        if out_nms["labels"][n] in [1]:
                            falha = 1 # muda o flag
                            sf = 1
                            col = (0,0,255)
                            tam = 100*(np.abs(x1-x0)*np.abs(y1-y0))/(300*150)
                        
                        if out_nms["labels"][n] in [3]:
                            falha = 1
                            ot = 1
                            col = (255,0,0)
                            tam = 100*(np.abs(x1-x0)*np.abs(y1-y0))/(300*150)

                        if falha == 1:
                            im = cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)  
                            out_dict[keys[k]+"_label_"+str(cont)] = f"{int(trinca)}{int(sf)}{int(ot)}" #np.array([trinca, sf])
                            out_dict[keys[k]+"_tamanho_"+str(cont)] = float(tam)
                            cont += 1
                
                if falha == 1:
                    im_b64 = base64.b64encode(im).decode('utf-8')
                    out_dict[keys[k]] = im_b64
        
        stop = time.time()

        time_postprocessing = stop - start

        del pred_model # apaga as predicoes (liberar GPU)
        out_dict['request_time'] = time_request
        out_dict['preprocessing_time'] = time_preprocessing
        out_dict['processing_time'] = time_processing
        out_dict['postprocessing_time'] = time_postprocessing
        
        encodedNumpyData = json.dumps(out_dict)

        return encodedNumpyData

if __name__=="__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=6000)
