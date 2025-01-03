#!/bin/bash

choice=$1

rm -rf main

case $choice in
    "reduction")
        nvcc -w -o main reduction.cu
        ;;
    "scan")
        nvcc -w -o main scan.cu
        ;;
    *)
        echo "error in choice"
        ;;
esac

./main