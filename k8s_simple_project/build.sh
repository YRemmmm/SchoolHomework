#!/bin/bash

sh delete.sh

###############################

cd gateway-custom
sh build.sh
cd ..

cd common-custom
sh build.sh
cd ..

cd user-custom
sh build.sh
cd ..

cd word-custom
sh build.sh
cd ..

###############################

rm -rf /mnt/data/*

mkdir -p /mnt/data/consul/
cp -f services.json /mnt/data/consul/
kubectl create -f consul-cluster1.yaml

mkdir -p /mnt/data/tomcat1/
cp -r ./webapps/* /mnt/data/tomcat1/
kubectl create -f tomcat-cluster1.yaml

mkdir -p /mnt/data/common1/
cp ./common-custom/Common-0.0.1-SNAPSHOT.war /mnt/data/common1/app.war
kubectl create -f common-cluster1.yaml

mkdir -p /mnt/data/user1/
cp ./user-custom/User-0.0.1-SNAPSHOT.war /mnt/data/user1/app.war
kubectl create -f user-cluster1.yaml

mkdir -p /mnt/data/word1/
cp ./word-custom/Word-0.0.1-SNAPSHOT.war /mnt/data/word1/app.war
kubectl create -f word-cluster1.yaml

kubectl create -f consul-gateway.yaml

kubectl create -f ingress-class.yaml

kubectl get po | grep common | awk '{print $1}' | xargs kubectl delete pod
kubectl get po | grep user | awk '{print $1}' | xargs kubectl delete pod
kubectl get po | grep word | awk '{print $1}' | xargs kubectl delete pod

sudo sh -c 'echo "192.168.137.2 test.sample.com" >> /etc/hosts'