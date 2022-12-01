FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN mkdir projeto

COPY requirements.txt ./requirements.txt

#COPY obj_detection.pt ./projeto/modelo_rcnn.ckpt

COPY app.py ./projeto/app.py

RUN apt-get update
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install -r requirements.txt 

EXPOSE 6000

CMD ["python", "./projeto/app.py"]