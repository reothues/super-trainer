# syntax=docker/dockerfile:1
FROM python:latest
WORKDIR /
COPY . .
RUN pip install -q tflite-model-maker; pip install numpy==1.23.4
CMD ["python", "main.py"]
