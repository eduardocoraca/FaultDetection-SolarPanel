FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir projeto

COPY requirements.txt ./requirements.txt
COPY obj_detection.pt ./projeto/modelo_rcnn.ckpt
COPY app.py ./projeto/app.py

RUN apt-get update
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install -r requirements.txt 

EXPOSE 6000

CMD ["python", "./projeto/app.py"]