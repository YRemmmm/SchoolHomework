#!/bin/bash

docker pull openjdk:17-jdk-alpine
docker rmi gateway:1.0
docker build -t gateway:1.0 .