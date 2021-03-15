FROM python:3.8.6-buster

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt setup.sh ./

# Install requirements.
# Remove the requirements.txt file after 
# install as the file will be needed no more
RUN pip install -r requirements.txt \
    && rm requirements.txt