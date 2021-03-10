FROM python:3.8.6-buster

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install autopep8==1.5.5