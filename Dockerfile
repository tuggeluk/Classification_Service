FROM ubuntu:latest
MAINTAINER Ange Gallego "ange.r.gallego@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get install -y python-imaging
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ./start_server.sh