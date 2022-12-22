import numpy as np
import cv2
from flask import Flask, request
import json
import base64

# funcoes necessarias para o recorte das celulas
def recortar(img, xc, yc, dx, dy, ex, ey):
    xidx = np.arange(xc - dx // 2 - ex, xc + dx // 2 + ex)
    yidx = np.arange(yc - dy // 2 - ey, yc + dy // 2 + ey)

    # flags to indicate if the cell is in the limit of the panel
    right = False
    left = False
    bottom = False
    top = False
    if xc + dx // 2 + ex >= img.shape[1]:
        xidx[xidx>=img.shape[1]] = img.shape[1] - (xidx[xidx>=img.shape[1]] - img.shape[1] + 1) # mirror
        right = True
    if xc - dx // 2 - ex < 0:
        xidx[xidx<0] = np.abs(xidx[xidx<0]) # mirror
        left = True
    if yc + dy // 2 + ey >= img.shape[0]:
        yidx[yidx>=img.shape[0]] = img.shape[0] - (yidx[yidx>=img.shape[0]] - img.shape[0] + 1) # mirror
        bottom = True
    if yc - dy // 2 - ey < 0:
        yidx[yidx<0] = np.abs(yidx[yidx<0]) # mirror
        top = True 

    out_img = img[yidx,:][:,xidx]
    if left:
        out_img[:,xidx==0] = 70    
    if right:
        out_img[:,xidx==img.shape[1]-1] = 70    
    if bottom:
        out_img[yidx==img.shape[0]-1,:] = 70  
    if top:
        out_img[yidx==0,:] = 70
        
    return out_img

def getCells(img):
    # funcao principal para recortar um painel em 72 celulas
    len_y, len_x = img.shape
    img = (img - img[0:int(0.75*len_y),:].min())/(img[0:int(0.75*len_y),:].max()-img[0:int(0.75*len_y),:].min()+1)
    img[img>1] = 1
    img[img<0] = 0
    img = np.asarray(255*img, dtype=np.uint8)

    img[img<30] = 0
    Nx = 24
    Ny = 6

    #x0 = np.where(np.abs(np.diff(img[int(0.25*len_y):int(0.75*len_y),:].mean(axis=0)))>1)[0][0]
    #x1 = np.where(np.abs(np.diff(img[int(0.25*len_y):int(0.75*len_y),:].mean(axis=0)))>1)[0][-1]

    #y0 = np.where(np.abs(np.diff(img[:,int(0.7*len_x):int(0.85*len_x)].mean(axis=1)))>10)[0][0]
    #y1 = np.where(np.abs(np.diff(img[:,int(0.7*len_x):int(0.85*len_x)].mean(axis=1)))>10)[0][-1]

    mean_x = img[:,int(0.7*len_x):int(0.85*len_x)].mean(axis=1)
    mean_y = img[int(0.25*len_y):int(0.75*len_y),:].mean(axis=0)

    diff_x = np.diff(mean_x)
    diff_y = np.diff(mean_y)

    x0 = np.argmax(diff_y[0:100])
    x1 = np.argmin(diff_y[len(diff_y)-150:]) + len(diff_y) - 150

    y0 = np.argmax(diff_x[0:100])
    y1 = np.argmin(diff_x[len(diff_x)-100:]) + len(diff_x) - 100

    wy = (y1-y0)//Ny
    wx = (x1-x0)//Nx

    img = img[y0:y1,:][:,x0:x1]
    len_y, len_x = img.shape
    xc = np.linspace((wx//2), len_x-(wx//2), Nx, dtype=int)
    yc = np.linspace((wy//2), len_y-(wy//2), Ny, dtype=int)    
    X, Y = np.meshgrid(xc, yc)
    ex = 30
    ey = 12
    if wx>100:
        out = {}
        out_1 = {}
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                dx = wx
                dy = wy
                rec0 = recortar(img, X[j, i], Y[j, i], dx, dy, 0, 0)
                if rec0.mean() > 30:
                    rec1 = recortar(img, X[j, i], Y[j, i], dx, dy, ex, ey)
                    rec2 = recortarRefinadoJanela(rec1, ex, ey)
                else:
                    rec2 = rec0
                coordY = (['24','23','22','21','20','19','18','17','16','15','14','13','12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1'])[i]
                coordX = np.flip(['A','B','C','D','E','F'])[j]
                rec2 = np.expand_dims(rec2, 2)
                rec2 = np.tile(rec2, (1,1,3))
                out[coordY + coordX] = rec2
        return out, [x0,y0,x1,y1]
    else:
        return False

def checar(path_dados, file):
	img = cv2.imread(path_dados + file, cv2.IMREAD_GRAYSCALE)
	bin = img.copy()
	t = img.mean() - 2 * img.std()
	bin[img < t] = 0
	bin[img >= t] = 1
	if (bin == 1).sum() == bin.shape[0] * bin.shape[1]:
		return False
	else:
		return True

def recortarRefinadoJanela(img, ex, ey):
    len_y, len_x = img.shape
    imgcopy = img.copy()
    imgcopy = (imgcopy - imgcopy.min())/(imgcopy.max()-imgcopy.min()+1)
    imgcopy = np.asarray(255*imgcopy, dtype=np.uint8)

    imgcopy[imgcopy<50] = 0
    blur = cv2.GaussianBlur(imgcopy, (5, 5), 0)
    imgcp = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    x_l = np.argmax(imgcp[int(0.25*len_y):int(0.75*len_y),:][:,0:2*ex].mean(axis=0))
    x_r = imgcp.shape[1] - (2*ex - np.argmin(imgcp[int(0.25*len_y):int(0.75*len_y),:][:,-2*ex:].mean(axis=0)))
    
    imgcp = cv2.Sobel(blur[:, x_l:x_r], cv2.CV_32F, 0, 1)
    p_max = np.argmax(imgcp[0:2*ey, int(0.25*len_x):int(0.75*len_x)].mean(axis=1))
    p_min = imgcp.shape[0] - (2*ey - np.argmin(imgcp[-2*ey:, int(0.25*len_x):int(0.75*len_x)].mean(axis=1)))
    y_l = p_max
    y_r = p_min
    return img[:, x_l:x_r][y_l:y_r, :]


# aplicativo flask
app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Cell split</h1>"

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        img_json = request.get_json()
        img_json = json.loads(img_json)
        size_x = int(img_json["size_x"])
        size_y = int(img_json["size_y"])

        x = np.frombuffer(base64.b64decode(img_json["img"]), dtype=np.uint8).reshape(size_y,size_x)
        #x = np.asarray(img_json["panel"], dtype=np.uint8)
        if len(x.shape)>2:
            x = x[:,:,0] # panel image must be a matrix
        
        del img_json # removing json from memory
        cells, coord = getCells(x) # obtaining cells

        # the output is a dict with the local as key and a 150x300 matrix
        out = {}
        for k in cells.keys():
            out_img = cv2.resize(cells[k][:,:,0], (150,300), interpolation=cv2.INTER_AREA)
            out_img = base64.b64encode(out_img).decode('utf-8')
            out[k] = out_img

        #response = json.dumps(out, cls=NumpyArrayEncoder)
        response = json.dumps(out)
        return response

if __name__=="__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=3000)