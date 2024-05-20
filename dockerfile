FROM python:3.10-slim

WORKDIR /app
COPY . /app
VOLUME ["/src", "/app/src"]

RUN apt-get update \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir torch torchaudio

EXPOSE 8000
ENV NAME env_file

WORKDIR /app/src
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--ssl-keyfile", "ssl/local_key.pem", "--ssl-certfile", "ssl/local_cert.pem"]
