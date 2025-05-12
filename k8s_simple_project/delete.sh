#!/bin/bash

kubectl delete -f consul-cluster1.yaml
kubectl delete -f tomcat-cluster1.yaml
kubectl delete -f consul-gateway.yaml
kubectl delete -f ingress-class.yaml
kubectl delete -f common-cluster1.yaml
kubectl delete -f user-cluster1.yaml
kubectl delete -f word-cluster1.yaml