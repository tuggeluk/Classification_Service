#!/bin/bash
docker build -t classifier-server:latest .
docker run -d -p 5000:5000 --net=host classifier-server
