FROM python:3.10

WORKDIR /app
COPY . /app
VOLUME ["/src", "/app/src"]

RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra build-essential
RUN python3 -m pip install torch torchvision torchaudio
RUN python3 -m pip install -r requirements.txt

EXPOSE 8000
ENV NAME env_file

WORKDIR /app/src
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
