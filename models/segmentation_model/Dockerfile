FROM tensorflow/tensorflow:2.10.0-gpu

RUN mkdir projeto

COPY requirements.txt ./requirements.txt

#COPY NP_Teste_modelo_Ucnn_map_102_model_005.h5 ./projeto/ucnn.h5
#COPY PN_Teste_modelo_Unet_map_102_classe_001_model_003.h5 ./projeto/unet.h5

COPY app.py ./projeto/app.py

RUN apt-get update
RUN apt-get install python3.8-venv -y
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt 

EXPOSE 4000

CMD ["python3", "./projeto/app.py"]