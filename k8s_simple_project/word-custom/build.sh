#!/bin/bash

docker pull tomcat:10.0.14
docker rmi word:1.0
docker build -t word:1.0 .