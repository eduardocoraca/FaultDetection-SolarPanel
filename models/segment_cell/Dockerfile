FROM python:3.7-buster

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir projeto

COPY requirements.txt ./requirements.txt
COPY app.py ./projeto/app.py

RUN apt-get update
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

EXPOSE 3000

CMD ["python", "./projeto/app.py"]