FROM python:3.8.6-buster

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt