FROM python:3.7-buster

WORKDIR /app

COPY requirements.txt .
COPY st_lib.py .
COPY streamlit_app.py .
COPY .streamlit /app/.streamlit
COPY last_file.py .

RUN apt-get update
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.runOnSave=true"]