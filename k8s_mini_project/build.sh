#!/bin/bash

sudo chronyc -a makestep

docker pull registry.cn-hangzhou.aliyuncs.com/google_containers/defaultbackend:1.4
docker pull registry.cn-hangzhou.aliyuncs.com/google_containers/nginx-ingress-controller:v1.1.0
docker pull docker.1ms.run/redis:5.0.9-alpine
docker pull docker.1ms.run/lengcz/tomcat:1.0

cd mysql-custom
bash build.sh
cd ..

kubectl create namespace cluster-system

kubectl create secret generic mysql-root-password --from-literal=password=123456
mkdir -p /mnt/data/mysql/
kubectl create -f mysql-alone.yaml

mkdir -p /mnt/data/tomcat1/
mkdir -p /mnt/data/tomcat2/
cp -r ./ROOT-cluster1/* /mnt/data/tomcat1/
cp -r ./ROOT-cluster2/* /mnt/data/tomcat2/
kubectl create -f tomcat-cluster1.yaml
kubectl create -f tomcat-cluster2.yaml

kubectl create namespace lishanbin-service
kubectl apply -f redis-cluster.yaml -n lishanbin-service

NAMESPACE="lishanbin-service"
APP_LABEL="app=redis-cluster"
NODES=$(kubectl get pods -n $NAMESPACE -l $APP_LABEL -o jsonpath='{range.items[*]}{.status.podIP}:6379 ')
echo "Redis nodes: $NODES"
REDIS_PASS="123456"
kubectl exec -it redis-cluster-0 -n $NAMESPACE -- sh -c "redis-cli -a $REDIS_PASS --cluster create ${NODES// / } --cluster-replicas 1"

kubectl create -f admin-user.yaml
kubectl create -f default-backend.yaml
kubectl create -f nginx-ingress-lb.yaml
kubectl create -f ingress-class.yaml

sudo sh -c 'echo "10.233.0.21 m1.sample.com" >> /etc/hosts'
sudo sh -c 'echo "10.233.0.22 m2.sample.com" >> /etc/hosts'
