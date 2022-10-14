FROM python:3.7-buster

WORKDIR /app

COPY requirements.txt .
COPY st_lib_manual.py ./st_lib.py
COPY streamlit_app_manual.py ./streamlit_app.py

RUN apt-get update
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

EXPOSE 8503

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8503"]