FROM python:3.7-buster

WORKDIR /app

COPY requirements.txt .
COPY streamlit_config.py .

RUN apt-get update
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

EXPOSE 8502

ENTRYPOINT ["streamlit", "run", "streamlit_config.py", "--server.port=8502"]