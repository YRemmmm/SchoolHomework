#!/bin/bash

docker pull tomcat:10.0.14
docker rmi user:1.0
docker build -t user:1.0 .