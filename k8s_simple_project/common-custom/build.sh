#!/bin/bash

docker pull tomcat:10.0.14
docker rmi common:1.0
docker build -t common:1.0 .