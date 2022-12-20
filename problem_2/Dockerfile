# For more information, please refer to https://aka.ms/vscode-docker-python
FROM continuumio/miniconda3:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/applications/image_classifier
COPY . .

RUN echo ls
RUN pip install -r requirements.txt
RUN mkdir dataset

ENTRYPOINT ["python", "src/train.py"]
