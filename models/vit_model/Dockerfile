FROM tensorflow/tensorflow:2.10.0-gpu

RUN mkdir projeto

COPY requirements.txt ./requirements.txt

COPY ViT_models ./projeto/ViT_models
COPY Flaw_detection_FEEC.py ./projeto/Flaw_detection_FEEC.py
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

EXPOSE 7000

CMD ["python3", "./projeto/app.py"]