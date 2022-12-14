import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from flask import Flask, request
import json
from PIL import Image
from json import JSONEncoder
import requests
import streamlit as st
import time
import io
import mysql.connector
from datetime import datetime
import base64
from typing import Tuple, Dict, List
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import last_file

# for automatic mode
class Watchdog(FileSystemEventHandler):
    def __init__(self, hook):
        self.hook = hook

    def on_modified(self, event):
        if (event.is_directory==False) & (event.src_path.endswith('.jpg')):
            self.hook(event.src_path)

def update_last_file(img_path):
    time.sleep(0.5)
    last_file_path = last_file.__file__
    with open(last_file_path, "w") as fp:
        fp.write(f'path = "{img_path}"')

# initialization
@st.experimental_memo(show_spinner=False)
def install_monitor():
    observer = Observer()
    observer.schedule(
        Watchdog(update_last_file),
        path="images_folder/",
        recursive=True)
    observer.start()

def reset():
    last_file_path = last_file.__file__
    st.session_state['uploaded_file'] = None
    with open(last_file_path, "w") as fp:
        fp.write(f'path = ""')  

@st.experimental_memo(show_spinner=False)
def set_layout_config():
    st.set_page_config(layout="wide")
    st.markdown("""
            <style>
                .css-18e3th9 {
                    padding: 2rem 5rem 0rem;
                }
                body{
                    font-size:0.75rem;
                }
                .css-hxt7ib{
                    padding-top: 2rem;
                }
            </style>
            """, 
            unsafe_allow_html=True)

    # hiding row index
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.markdown(
        """<style>
            .table {text-align: left !important}
        </style>
        """, unsafe_allow_html=True) 

@st.experimental_memo(show_spinner=False)
def initialize():
    st.session_state['img_path'] = ''
    st.session_state['uploaded_file'] = None

def update_criteria():
    '''Updates the criteria according to the config.ym. file
    '''
    with open('data/config.yml', 'r') as c:
        config = yaml.safe_load(c)
    st.session_state['criterio_tr'] = config['criterios']['trinca']
    st.session_state['criterio_sf'] = config['criterios']['solda_fria']

# password
def logout():
    st.session_state["password"] = ""
    st.session_state["password_correct"] = False

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    else:
        # Password correct.
        return True

# function to convert np.array to binary
def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()

# models
@st.experimental_memo(show_spinner=False)
def request_cell_split(img: np.array) -> dict:
    ''' Sends panel image to 'segment_cell' model
    Args:
        img: (img_x, img_y) np.array with dtype=np.uint8 
    Returns:
        dict: dict of cells, with keys relrated to each cell position. Eg. 1A,...,24F
    '''
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_b64 = base64.b64encode(img).decode("utf-8")
    size_y, size_x = img.shape

    request_file = {"img": img_b64, "size_x": size_x, "size_y": size_y}
    request_file = json.dumps(request_file)

    ip = "segment_cell" # ip as the name of the docker container, which must be at port 3000
    #ip = "localhost"
    response = requests.post("http://" + ip + ":3000/predict/", json=request_file)
    json_resp = json.loads(response.text)
    cells = {}
    for k in json_resp.keys():
        img_resp = np.frombuffer(base64.b64decode(json_resp[k]), dtype=np.uint8).reshape(300,150)
        cells[k] = img_resp # the output is a dict with the local as key and a 150x300 matrix

    del json_resp # remove response JSON from memory

    # construcao da imagem do painel corrigida, considerando cada celula 300x150
    img_painel = np.zeros((300*6, 150*24), dtype=np.uint8)
    img_painel = np.expand_dims(img_painel, -1)
    img_painel = np.tile(img_painel, (1,1,3))

    end = time.time()    
    print(f"Tempo para recorte: {end-start}s.")

    # pd.DataFrame containing metadata from the model
    df_meta = pd.DataFrame()
    df_meta['model'] = ['Recorte']
    df_meta['t_request'] = [0]
    df_meta['t_processing'] = [0]
    df_meta['t_processing'] = [0]
    df_meta['t_postprocessing'] = [0]
    df_meta['processed_cells'] = [0]
    df_meta['t_total'] = [end-start]

    return cells, img_painel, df_meta

@st.experimental_memo(show_spinner=False)
def get_interactive_cells(cells: dict, df: pd.DataFrame) -> str:
    ''' Create an interactive array of cells in html format.
    Args:
        cells: dict of grayscale image of cells containing np.arrays with dtype=np.uint8
        df: pd.DataFrame containing the predictions for each cell
    Returns:
        str: html for building the entire interactive panel
    '''
    content = ""

    # first row
    for i in range(25):
        img_text = 255*np.ones((100,150), dtype=np.uint8)
        if i!=0:
            text = f"{i}"
        else:
            text = ""
        img_text = cv2.putText(img=img_text, text=text, org=(50,75), 
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0), thickness=4)
        image_cv = cv2.cvtColor(img_text.copy(), cv2.COLOR_RGB2BGR)
        _,buffer = cv2.imencode('.jpeg', image_cv)
        encoded = base64.b64encode(buffer).decode()     
        content += f"""<img width='3.75%' src='data:image/jpeg;base64,{encoded}'></a>"""
    content += "<br>"

    for j in range(6):
        cy = ["A","B","C","D","E","F"][j]
        img_text = 255*np.ones((100,150), dtype=np.uint8)
        img_text = cv2.putText(img=img_text, text=f"{cy}", org=(75,75), 
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0), thickness=4)
        image_cv = cv2.cvtColor(img_text.copy(), cv2.COLOR_RGB2BGR)
        _,buffer = cv2.imencode('.jpeg', image_cv)
        encoded = base64.b64encode(buffer).decode()     
        content += f"""<img width='3.75%' src='data:image/jpeg;base64,{encoded}'></a>"""
        # content += f """<b>{cy}  </b>"""
        for i in range(24):
            cx = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"][i]
            img_celula = cells[cx+cy].copy()
            img_celula = cv2.resize(img_celula, (150,300), cv2.INTER_AREA)

            if cx+cy in np.unique(df["Celula"]):
                img_celula = np.expand_dims(img_celula,-1)
                img_celula = np.tile(img_celula, (1,1,3))
                img_celula = cv2.rectangle(img_celula, (0,0), (150,300), (255,0,0), 30)

            image_cv = cv2.cvtColor(img_celula.copy(), cv2.COLOR_RGB2BGR)
            _,buffer = cv2.imencode('.jpeg', image_cv)
            encoded = base64.b64encode(buffer).decode()
            content += f"""<a href='#' id='{cx}{cy}'><img width='3.75%' src='data:image/jpeg;base64,{encoded}'></a>"""
        content += "<br>"
    return content

@st.experimental_memo(show_spinner=False)
def request_pred_detection(cells: dict) -> Tuple[pd.DataFrame, dict]:
    '''Sends cells for 'detection_model'.
    Args:
        cells: dict of cells
    Returns:
        df: pd.DataFrame containing the predictions for each cell
        cells_fault: dict of cells with predictions (bounding boxes)
    '''
    start = time.time()
    cells_dict = {}
    for k in cells.keys():
        img = cells[k] # each image is a 150x300 matrix
        img = base64.b64encode(img).decode('utf-8')
        cells_dict[k] = img

    img_json = json.dumps(cells_dict)
    ip = "detection_model"
    #ip = "localhost"
    response = requests.post("http://" + ip + ":6000/predict/", json=img_json)
    json_resp = json.loads(response.text)

    chaves = list(json_resp.keys())
    outros = list(filter(lambda x: x if len(x.split("_")) > 1 else None, chaves))
    labels = list(filter(lambda x: x if x.split("_")[1]=="label" else None, outros))
    tamanho = list(filter(lambda x: x if x.split("_")[1]=="tamanho" else None, outros))
    imagens = list(filter(lambda x: x if len(x.split("_")) == 1 else None, chaves))

    start_post = time.time()
    results_list = []
    cells_fault = {}
    for k in imagens:
        out_detec = np.frombuffer(base64.b64decode(json_resp[k]), dtype=np.uint8).reshape(300,150,3)
        num_falhas = len(list(filter(lambda x: x.startswith(k + "_label_"), labels))) # faults detected at cell k
        for n in range(num_falhas): # for each fault in the cell
            pred_str = json_resp[k+"_label_"+str(n)]
            if pred_str[0]=="1":
                pred = 1
            elif pred_str[1]=="1":
                pred = 0

            status = ["Solda fria", "Trinca"][pred]
            tamanho = np.array(json_resp[k+"_tamanho_"+str(n)]).squeeze()
            res = {
                'Celula': k,
                'Modelo': 'Detec????o',
                'Status': status,
                'Tamanho': np.round(tamanho,2),
                'Unidade': '%',
                }
            results_list.append(res)
        if num_falhas>0: # saving only the images with a detected fault
            cells_fault[k] = out_detec
    end_post = time.time()
    df = pd.DataFrame(results_list)
    end = time.time()

    # pd.DataFrame containing metadata from the model
    df_meta = pd.DataFrame()
    df_meta['model'] = ['Detec????o']
    df_meta['t_request'] = [float(json_resp['request_time'])]
    df_meta['t_preprocessing'] = [float(json_resp['preprocessing_time'])]
    df_meta['t_processing'] = [float(json_resp['processing_time'])]
    df_meta['t_postprocessing'] = [float(json_resp['postprocessing_time']) + (end_post-start_post)]
    df_meta['processed_cells'] = [len(list(cells_fault.keys()))]
    df_meta['t_total'] = [end-start]

    return df, cells_fault, df_meta

@st.experimental_memo(show_spinner=False)
def request_pred_segmentation(cells: dict) -> Tuple[pd.DataFrame, dict]:
    '''Sends cells for 'segmentation_model'.
    Args:
        cells: dict of cells
    Returns:
        df: pd.DataFrame containing the predictions for each cell
        cells_fault: dict of cells with predictions (pixel-level segmentation)
    '''
    start = time.time()
    cells_dict = {}
    for k in cells.keys():
        img = cells[k] # each image is a 150x300 matrix
        img = base64.b64encode(img).decode('utf-8')
        cells_dict[k] = img

    img_json = json.dumps(cells_dict)
    ip = "segmentation_model"
    #ip = "localhost"
    response = requests.post("http://" + ip + ":4000/predict/", json=img_json)
    json_resp = json.loads(response.text)

    chaves = list(json_resp.keys())
    outros = list(filter(lambda x: x if len(x.split("_")) > 1 else None, chaves))
    labels = list(filter(lambda x: x if x.split("_")[1]=="label" else None, outros))
    #tamanho = list(filter(lambda x: x if x.split("_")[1]=="tamanho" else None, outros))
    imagens = list(filter(lambda x: x if len(x.split("_")) == 1 else None, chaves))

    results_list = []
    cells_fault = {}
    for k in imagens:
        out_detec = np.frombuffer(base64.b64decode(json_resp[k]), dtype=np.uint8).reshape(300,150,3)
        num_falhas = len(list(filter(lambda x: x.startswith(k + "_label_"), labels))) # faults detected at cell k
        for n in range(num_falhas): # for each fault in the cell
            pred_str = json_resp[k+"_label_"+str(n)]
            if pred_str[0]=="1":
                status = "Trinca"
            elif pred_str[1]=="1":
                status = "Solda fria"
            elif pred_str[2]=="1":
               status = "Outros"

            tamanho = np.array(json_resp[k+"_tamanho_"+str(n)]).squeeze()
            res = {
                'Celula': k,
                'Modelo': 'Segmenta????o',
                'Status': status,
                'Tamanho': np.round(tamanho,2),
                'Unidade': '%',
                    }
            results_list.append(res)
        if num_falhas>0: # saving only the images with a detected fault
            original = cells[k]
            original = cv2.cvtColor(original,cv2.COLOR_GRAY2RGB)
            im_add = cv2.addWeighted(out_detec, 0.5, original , 0.5, 0.0)
            cells_fault[k] = im_add
    
    df = pd.DataFrame(results_list)
    end = time.time()
    print(f"Tempo para segmenta????o: {end-start}s.")

    # pd.DataFrame containing metadata from the model
    df_meta = pd.DataFrame()
    df_meta['model'] = ['Segmenta????o']
    df_meta['t_request'] = [float(json_resp['request_time'])]
    df_meta['t_preprocessing'] = [float(json_resp['preprocessing_time'])]
    df_meta['t_processing'] = [float(json_resp['processing_time'])]
    df_meta['t_postprocessing'] = [float(json_resp['postprocessing_time'])]
    df_meta['processed_cells'] = [len(chaves)]
    df_meta['t_total'] = [end-start]
    return df, cells_fault, df_meta

@st.experimental_memo(show_spinner=False)
def request_pred_vit(cells: dict) -> Tuple[pd.DataFrame, dict]:
    '''Sends cells for 'vit_model'.
    Args:
        cells: dict of cells
    Returns:
        df: pd.DataFrame containing the predictions for each cell
        cells_fault: dict of cells with predictions (pixel-level attention map)
    '''
    start = time.time()
    cells_dict = {}
    for k in cells.keys():
        img = cells[k] # each image is a 150x300 matrix
        img = cv2.resize(img, (150,280), cv2.INTER_AREA) # converting to 150x280 for ViT
        img = base64.b64encode(img).decode('utf-8')
        cells_dict[k] = img

    img_json = json.dumps(cells_dict)
    ip = "vit_model"
    response = requests.post("http://" + ip + ":7000/predict/", json=img_json)
    json_resp = json.loads(response.text)

    chaves = list(json_resp.keys())
    outros = list(filter(lambda x: x if len(x.split("_")) > 1 else None, chaves))
    labels = list(filter(lambda x: x if x.split("_")[1]=="label" else None, outros))
    imagens = list(filter(lambda x: x if len(x.split("_")) == 1 else None, chaves))

    results_list = []
    cells_fault = {}
    for k in imagens:
        out_detec = np.frombuffer(base64.b64decode(json_resp[k]), dtype=np.uint8).reshape(280,150)
        out_detec = cv2.resize(out_detec, (150,300))
        out_detec = np.expand_dims(out_detec,-1)
        out_detec = np.tile(out_detec, (1,1,3))
        num_falhas = len(list(filter(lambda x: x.startswith(k + "_label_"), labels))) # faults detected at cell k
        tamanhos = []
        for n in range(num_falhas): # for each fault in the cell
            pred_str = json_resp[k+"_label_"+str(n)]
            if pred_str[0]=="1":
                status = "Trinca"
            elif pred_str[1]=="1":
                status = "Solda fria"
            elif pred_str[2]=="1":
               status = "Outros"

            tamanho = np.array(json_resp[k+"_tamanho_"+str(n)]).squeeze()
            tamanhos.append(tamanho)
            if tamanho > 0:
                res = {
                    'Celula': k,
                    'Modelo': 'Vision Transformer',
                    'Status': status,
                    'Tamanho': np.round(tamanho,2),
                    'Unidade': '%',
                        }
                results_list.append(res)

        if np.any(np.array(tamanhos))>0: # saving only the images with a detected fault
            original = cells[k]
            original = cv2.cvtColor(original,cv2.COLOR_GRAY2RGB)
            im_add = cv2.addWeighted(out_detec, 0.5, original , 0.5, 0.0)
            cells_fault[k] = im_add
    
    df = pd.DataFrame(results_list)
    end = time.time()
    print(f"Tempo para predi????o: {end-start}s.")

    # pd.DataFrame containing metadata from the model
    df_meta = pd.DataFrame()
    df_meta['model'] = ['Vision Transformer']
    df_meta['t_request'] = [float(json_resp['request_time'])]
    df_meta['t_preprocessing'] = [float(json_resp['preprocessing_time'])]
    df_meta['t_processing'] = [float(json_resp['processing_time'])]
    df_meta['t_postprocessing'] = [float(json_resp['postprocessing_time'])]
    df_meta['processed_cells'] = [len(chaves)]
    df_meta['t_total'] = [end-start]
    return df, cells_fault, df_meta

@st.experimental_memo(show_spinner=False)
def get_image(uploaded_file) -> Tuple[np.array, str]:
    '''Opens the image sent via Streamlit.
    Args:
        uploaded_file: BytesIO.UploadedFile file sent via Streamlit
    Returns:
        image: np.array with 3 channels
        filename: str
    '''
    filename = uploaded_file.name
    image = Image.open(uploaded_file)
    image = np.array(image)
    return image, filename

@st.experimental_memo(show_spinner=False)
def get_image_auto(image_path:str) -> Tuple[np.array, str]:
    '''Opens the image from the image path.
    Args:
        image_path: path to the image in 'image_folders'
    Returns:
        image: np.array with 3 channels
        filename: str
    '''
    filename = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    return image, filename

# logging
@st.experimental_memo(show_spinner=False)
def log_results(path: str,
               filename: str,
               result: str,
               predictions: pd.DataFrame,
               crit_sf: float,
               crit_tr: float,
               pred_detection: pd.DataFrame,
               pred_segmentation: pd.DataFrame,
               pred_vit: pd.DataFrame,
               t_detection: float,
               t_segmentation: float,
               t_vit: float,
            ) -> None:
    ''' Writes data to csv file.
    Args:
        path: path to the csv file
        filename: panel identifier
        results: overall panel result ('NG' or 'OK')
        predictions: dataframe with predictions per cell
        crit_sf: SF criteria
        crit_tr: TR criteria
        pred_detection: results from detection model
        pred_segmentation: results from segmentation model
        pred_vit: results from VIT model
    '''
    with open('data/log_config.yml', 'r') as c:
        log_config = yaml.safe_load(c)
    date = datetime.now().strftime("%Y%m%d") # current date to be used as the CSV filename
    path = log_config['path']

    ### panels CSV
    num_celulas_ng = len(predictions.index)
    if result == "Painel NG":
        status = 1
    else:
        status = 0
    
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    painel = filename
    if (num_celulas_ng>0):
        with open(f'{path}panels_{date}.csv', 'a') as f:
            f.write(f'{painel},{status},{num_celulas_ng},{data_hora},{crit_sf},{crit_tr}\n')

    ### cells CSV
    if num_celulas_ng > 0:
        for k in np.unique(predictions["Celula"]):
            local = k
            painel = filename
            status_k = np.unique(predictions.loc[predictions["Celula"]==k, "Status"].item().split(',')) # falhas presentes na celula
            trinca = 0
            solda_fria = 0
            outros = 0
            if "Trinca" in status_k:
                trinca = 1
            if "Solda fria" in status_k:
                solda_fria = 1
            if "Outros" in status_k:
                outros = 1
            
            with open(f'{path}cells_{date}.csv', 'a') as f:
                f.write(f'{local},{painel},{trinca},{solda_fria},{outros}\n')            

    ### cells_detection CSV
    for idx, row in pred_detection.iterrows():
        local = row['Celula']
        painel = filename
        status_k = row['Status']
        tamanho_k = row['Tamanho']
        tempo_k = t_detection
        
        with open(f'{path}cells_detection_{date}.csv', 'a') as f:
            f.write(f'{local},{painel},{status_k},{tamanho_k},{tempo_k}\n') 

    ### cells_segmentation CSV
    for idx, row in pred_segmentation.iterrows():
        local = row['Celula']
        painel = filename
        status_k = row['Status']
        tamanho_k = row['Tamanho']
        tempo_k = t_segmentation
        with open(f'{path}cells_segmentation_{date}.csv', 'a') as f:
            f.write(f'{local},{painel},{status_k},{tamanho_k},{tempo_k}\n') 

    ### cells_vit CSV
    for idx, row in pred_vit.iterrows():
        local = row['Celula']
        painel = filename
        status_k = row['Status']
        tamanho_k = row['Tamanho']
        tempo_k = t_vit
        with open(f'{path}cells_vit_{date}.csv', 'a') as f:
            f.write(f'{local},{painel},{status_k},{tamanho_k},{tempo_k}\n') 

def save_to_db(img_panel: np.array,
               filename: str,
               result: str,
               predictions: pd.DataFrame,
               comments: str,
               crit_sf: float,
               crit_tr: float,
               pred_detection: pd.DataFrame,
               pred_segmentation: pd.DataFrame,
               pred_vit: pd.DataFrame,
               t_detection: float,
               t_segmentation: float,
               t_vit: float,
            ) -> None:
    ''' Writes data to MySQL DB.
    Args:
        img_panel: np.array containing panel image
        filename: panel identifier
        results: overall panel result ('NG' or 'OK')
        predictions: dataframe with predictions per cell
        comments: comments sent by the user
        crit_sf: SF criteria
        crit_tr: TR criteria
        pred_detection: results from detection model
        pred_segmentation: results from segmentation model
        pred_vit: results from VIT model
    '''

    ### loading config 
    with open('data/mysql_config.yml', 'r') as c:
        mysql_config = yaml.safe_load(c)

    ### panels table
    img_binary = numpy_to_binary(img_panel)

    num_celulas_ng = len(predictions.index)
    if result == "Painel NG":
        status = 1
    else:
        status = 0
    
    if (num_celulas_ng>0) | (comments != ""):
        data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        painel = filename
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO paineis (painel, status, img, num_celulas_ng, data_hora, comentarios, crit_sf, crit_tr) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (painel, status, img_binary, num_celulas_ng, data_hora, comments, crit_sf, crit_tr))
        
        connection.commit()
        last_id = cursor.lastrowid # valor unico inserido para este painel
        cursor.close()
        connection.close()

    ### tabela celulas
    if num_celulas_ng > 0:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        valores_a_inserir = [] # lista onde serao armazendos os valores (tuplas) a serem inseridos
        for k in np.unique(predictions["Celula"]):
            local = k
            painel = filename
            id_painel = last_id
            status_k = np.unique(predictions.loc[predictions["Celula"]==k, "Status"].item().split(',')) # falhas presentes na celula
            
            trinca = 0
            solda_fria = 0
            outros = 0
            if "Trinca" in status_k:
                trinca = 1
            if "Solda fria" in status_k:
                solda_fria = 1
            if "Outros" in status_k:
                outros = 1
            
            valores_a_inserir.append((id_painel, local, painel, trinca, solda_fria, outros))
        
    cursor.executemany('INSERT INTO celulas (id_painel, local, painel, trinca, solda_fria, outros) VALUES (%s, %s, %s, %s, %s, %s)', valores_a_inserir)    
    connection.commit()
    cursor.close()
    connection.close()

    ### tabela celulas deteccao
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    valores_a_inserir = [] # lista onde serao armazendos os valores (tuplas) a serem inseridos
    for idx, row in pred_detection.iterrows():
        local = row['Celula']
        painel = filename
        id_painel = last_id
        status_k = row['Status']
        tamanho_k = row['Tamanho']
        tempo_k = t_detection
            
        valores_a_inserir.append((id_painel, local, painel, status_k, tamanho_k, tempo_k))
        
    cursor.executemany('INSERT INTO celulas_deteccao (id_painel, local, painel, status, tamanho, tempo) VALUES (%s, %s, %s, %s, %s, %s)', valores_a_inserir)    
    connection.commit()
    cursor.close()
    connection.close()

    ### tabela celulas segmentacao
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    valores_a_inserir = [] # lista onde serao armazendos os valores (tuplas) a serem inseridos
    for idx, row in pred_segmentation.iterrows():
        local = row['Celula']
        painel = filename
        id_painel = last_id
        status_k = row['Status']
        tamanho_k = row['Tamanho']
        tempo_k = t_segmentation
        valores_a_inserir.append((id_painel, local, painel, status_k, tamanho_k, tempo_k))
        
    cursor.executemany('INSERT INTO celulas_segmentacao (id_painel, local, painel, status, tamanho, tempo) VALUES (%s, %s, %s, %s, %s, %s)', valores_a_inserir)    
    connection.commit()
    cursor.close()
    connection.close()

    ### tabela celulas vit
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    valores_a_inserir = [] # lista onde serao armazendos os valores (tuplas) a serem inseridos
    for idx, row in pred_vit.iterrows():
        local = row['Celula']
        painel = filename
        id_painel = last_id
        status_k = row['Status']
        tamanho_k = row['Tamanho']
        tempo_k = t_vit
        valores_a_inserir.append((id_painel, local, painel, status_k, tamanho_k, tempo_k))
        
    cursor.executemany('INSERT INTO celulas_vit (id_painel, local, painel, status, tamanho, tempo) VALUES (%s, %s, %s, %s, %s, %s)', valores_a_inserir)    
    connection.commit()
    cursor.close()
    connection.close()