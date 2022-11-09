import streamlit as st
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from st_lib import *
from st_click_detector import click_detector
import mysql.connector
import base64
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import last_file

### initialization
set_layout_config() # sets layout markdown (oly 1st run)
initialize()  # sets 'manual' mode (only 1st run)
install_monitor() # starts monitoring (only 1st run)
update_criteria() # updates criteria from .yml file

resultado = ""
st.sidebar.subheader("Envio de imagem")
st.sidebar.write('Modo de operação: Automático')
st.subheader("Detecção de falhas em painéis fotovoltaicos")

image_path = last_file.path # path to the most recent image

### Main
if len(image_path)==0:
    pass
else:
    image, filename = get_image_auto(image_path)
    st.sidebar.subheader("Painel selecionado")
    st.sidebar.text(f"Painel: {filename.split('.')[0]}")

    ## Criteria
    criterio_sf = st.session_state["criterio_sf"]
    criterio_tr = st.session_state["criterio_tr"]
    criterio_ot = st.session_state["criterio_ot"]

    st.sidebar.text(f"Critério TR: >{criterio_tr}%")
    st.sidebar.text(f"Critério SF: >{criterio_sf}%")
    st.sidebar.text(f"Critério OT: >{criterio_ot}%")

    ## Request: cell segmentation
    cells, img_painel, meta_split = request_cell_split(image) # cells: dict with the local as key and a 150x300 matrix

    ## Request: detection
    pred_detection, cells_detection, meta_detection = request_pred_detection(cells)

    ## Request: segmentation model
    pred_segmentation, cells_segmentation, meta_segmentation = request_pred_segmentation(cells)

    ## Request: ViT model
    pred_vit, cells_vit, meta_vit = request_pred_vit(cells)

    ## Results Dataframes
    pred = pd.concat((pred_detection, pred_segmentation, pred_vit)).sort_values(by="Celula", ascending=True)
    pred_sem_vit = pred[pred['Modelo'] != 'Vision Transformer']    
    #pred_sem_vit = pd.concat((pred_detection, pred_segmentation)).sort_values(by="Celula", ascending=True)
    
    pred = pred.reset_index(drop=True)
    pred_sem_vit = pred_sem_vit.reset_index(drop=True)

    meta = pd.concat((meta_split, meta_detection, meta_segmentation, meta_vit))
    meta = meta.reset_index(drop=True)

    if len(pred.index)>0:
        resultado = "Painel NG"
    else:
        resultado = "Painel OK"


    ## cell report dataframe
    results_dict = {
        'Celula': [],
        'Status': []
    }
    # for each cell: check if sum of lengths of each flaw is 
    # larger than the threshold
    num_ng_cells = 0 # counter: number of NG cells
    for c in list(np.unique(pred_sem_vit['Celula'])):
        #f = pred_sem_vit.loc[((pred_sem_vit['Celula']==c) & (pred_sem_vit['Tamanho']>criterio_tr) & (pred_sem_vit['Status']=='Trinca')) | ((pred_sem_vit['Celula']==c) & (pred_sem_vit['Tamanho']>criterio_sf) & (pred_sem_vit['Status']=='Solda fria')), 'Status']
        f_tr = pred_sem_vit.loc[((pred_sem_vit['Celula']==c) & (pred_sem_vit['Status']=='Trinca'))]
        f_sf = pred_sem_vit.loc[((pred_sem_vit['Celula']==c) & (pred_sem_vit['Status']=='Solda fria'))]
        f_ot = pred_sem_vit.loc[((pred_sem_vit['Celula']==c) & (pred_sem_vit['Status']=='Outros'))]

        size_tr_total = np.max([np.sum(f_tr.loc[f_tr['Modelo']==model, 'Tamanho']) for model in np.unique(f_tr['Modelo'])] + [0] )
        size_sf_total = np.max([np.sum(f_sf.loc[f_sf['Modelo']==model, 'Tamanho']) for model in np.unique(f_sf['Modelo'])] + [0] )
        size_ot_total = np.max([np.sum(f_ot.loc[f_ot['Modelo']==model, 'Tamanho']) for model in np.unique(f_ot['Modelo'])] + [0] )
        
        falhas = ''
        outros = ''
        if size_tr_total > criterio_tr:
            falhas += 'Trinca,'
        if size_sf_total > criterio_sf:
            falhas += 'Solda fria,'
        if size_ot_total > criterio_ot:
            outros += 'Outros,'
        if len(falhas) > 0:
            num_ng_cells += 1
            results_dict['Celula'].append(c)
            results_dict['Status'].append(falhas + outros)

    results_df = pd.DataFrame()
    results_df['Celula'] = results_dict['Celula']
    results_df['Status'] = results_dict['Status']
    results_df.sort_values(by="Celula", ascending=True)

    ## Results
    st.sidebar.text(f"Resultado: {resultado}")
    st.sidebar.text(f"Células NG: {num_ng_cells}")

    ## Automatic log to DB
    save_to_db(
                img_panel=image,
                filename=filename,
                result=resultado,
                predictions=results_df,
                num_ng_cells = num_ng_cells,
                comments='',
                crit_sf=criterio_sf,
                crit_tr=criterio_tr, 
                pred_detection=pred_detection, 
                pred_segmentation=pred_segmentation, 
                pred_vit=pred_vit,
                t_detection=meta_detection['t_total'].item(),
                t_segmentation=meta_segmentation['t_total'].item(),
                t_vit=meta_vit['t_total'].item()
                )

    ## Automatic log to CSV
    log_results(
            path='./output_folder/',
            image_path=image_path,
            filename=filename,
            result=resultado,
            predictions=results_df,
            num_ng_cells=num_ng_cells,
            crit_sf=criterio_sf,
            crit_tr=criterio_tr, 
            pred_detection=pred_detection, 
            pred_segmentation=pred_segmentation, 
            pred_vit=pred_vit,
            t_detection=meta_detection['t_total'].item(),
            t_segmentation=meta_segmentation['t_total'].item(),
            t_vit=meta_vit['t_total'].item()
    )

    ## Panel report 
    st.sidebar.table(results_df)

    col_fig1, col_fig2, col_fig3, col_fig4, col_fig5 = st.columns([10,1.75,1.75,1.75,1.75])

    ## Interactive image
    content = get_interactive_cells(cells, results_df)

    st.write("Clique na célula de interesse para visualizar os resultados. As células em vermelho foram consideradas NG.")
    with col_fig1:
        clicked = click_detector(content)
        if clicked != "":
            selected_cell = clicked
        else:
            selected_cell = list(cells.keys())[0]

    ## cell result tab
    if selected_cell!="":
        k = selected_cell

        fig_cell = plt.figure()
        ax = fig_cell.add_subplot()
        ax.axis("off")
        ax.imshow(cells[k], cmap="gray")
        ax.set_title(f"Célula {k}")

        try:
            fig_cell_detect = plt.figure()
            ax1 = fig_cell_detect.add_subplot()
            ax1.imshow(cells_detection[k])
            ax1.set_title("Detecção")
            ax1.axis("off")
            with col_fig3:
                st.pyplot(fig_cell_detect)
        except:
            pass

        try:
            fig_cell_segment = plt.figure()
            ax1 = fig_cell_segment.add_subplot()
            ax1.imshow(cells_segmentation[k])
            ax1.set_title("Segmentação")
            ax1.axis("off")
            with col_fig4:
                st.pyplot(fig_cell_segment)
        except:
            pass

        try:
            fig_cell_vit = plt.figure()
            ax1 = fig_cell_vit.add_subplot()
            ax1.imshow(cells_vit[k])
            ax1.set_title("Vision Transformer")
            ax1.axis("off")
            with col_fig5:
                st.pyplot(fig_cell_vit)
        except:
            pass    

        with col_fig2:
            st.pyplot(fig_cell)

    plt.close("all")

    ## report of the selected cell
    st.write(f"Relatório da célula selecionada ({k})")
    df = pred.loc[pred["Celula"]==k,].sort_values('Modelo')
    df['Tamanho'] = df['Tamanho'].astype(str)

    st.table(df)   

    ## Execution details and configuration
    col_exec, col_conf = st.columns([1,1])
    with col_exec:
        with st.expander('Detalhes de execução'):
            meta_show = meta[['model','t_total']]
            meta_show.columns = ['Modelo','Tempo de execução (s)']
            t_total = np.round(meta_show["Tempo de execução (s)"].sum(),1)
            meta_show['Tempo de execução (s)'] = meta_show['Tempo de execução (s)'].copy().astype(str)
            st.write(f'Tempo total: {t_total} segundos.')
            st.table(meta_show)

    
