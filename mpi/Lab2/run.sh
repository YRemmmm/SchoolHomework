#!/bin/bash

# export PATH=/home/yr/Public/mpi/openmpi/bin/:$PATH
# export PATH=/home/yr/Public/mpi/mpich/bin/:$PATH
# export PMIX_MCA_pcompress_base_silence_warning=1
choice=$1
# read -p "Please select an option(basic, red, mpibasic, mpired)" choice

rm -rf nbody_basic nbody_red pth_nbody_basic pth_nbody_red

case $choice in
    "basic")
        gcc -pg -g -Wall -o nbody_basic nbody_basic.c -lm
        rm -rf output
        ./nbody_basic 5000 500 0.01 10 g 2>&1 | tee result/basic
        gprof ./nbody_basic gmon.out > profiling.txt
        echo "Run nbody_basic in serial mode."
        ;;
    "red")
        gcc -pg -g -Wall -o nbody_red nbody_red.c -lm
        rm -rf output
        ./nbody_red 5000 500 0.01 10 g 2>&1 | tee result/red
        gprof ./nbody_red gmon.out > profiling.txt
        echo "Run nbody_red in serial mode."
        ;;
    "pthbasic")
        gcc -pg -g -Wall -o pth_nbody_basic pth_nbody_basic.c -lm -lpthread 
        rm -rf output
        ./pth_nbody_basic 5000 500 0.01 10 g 2>&1 | tee output
        gprof ./pth_nbody_basic gmon.out > profiling.txt
        echo "Run nbody_basic in parallel mode."
        sh verify.sh basic
        ;;
    "pthred")
        gcc -pg -g -Wall -o pth_nbody_red pth_nbody_red.c -lm -lpthread 
        rm -rf output
        ./pth_nbody_red 5000 500 0.01 10 g 2>&1 | tee output
        gprof ./pth_nbody_red gmon.out > profiling.txt
        echo "Run nbody_red in parallel mode."
        sh verify.sh red
        ;;
    *)
        echo "Please select an option in: basic, red, pthbasic, pthred"
        ;;
esac
# --use-hwthread-cpus

