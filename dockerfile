FROM python:3.10-slim

WORKDIR /app
COPY . /app
VOLUME ["/src", "/app/src"]

RUN apt-get update && apt-get install -y build-essential ffmpeg \
    && python3 -m pip install --default-timeout=100 --no-cache-dir -r requirements.txt \
    && pip3 install --default-timeout=100 --no-cache-dir torch torchaudio

EXPOSE 8000
ENV NAME env_file

WORKDIR /app/src
CMD ["python3", "main.py"]
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--ssl-keyfile", "ssl/local_key.pem", "--ssl-certfile", "ssl/local_cert.pem"]
