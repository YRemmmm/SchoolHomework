#!/bin/bash


rm -rf main

choice=$1
case $choice in
    "csr")
        nvcc -w -o main spmv_csr.cu
        ;;
    "ell")
        nvcc -w -o main spmv_ell.cu
        ;;
    "merge")
        nvcc -w -o main merge.cu
        ;;
    *)
        echo "error choice"
        ;;
esac


choice=$2
case $choice in
    "log")
        ./main  2>&1 | tee output
        ;;
    *)
        ./main
        ;;
esac
