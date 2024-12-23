#!/bin/bash

# gcc -g -Wall -o nbody_basic nbody_basic.c -lm
# rm -rf basic_output
# ./nbody_basic 1000 4000 0.01 10 g 2>&1 | tee output

choice=$1
# read -p "Please select an option(basic, red, mpibasic, mpired)" choice

case $choice in
    "basic")
        echo "Run nbody_basic in serial mode."
        gcc -pg -g -Wall -o nbody_basic nbody_basic.c -lm
        rm -rf output
        ./nbody_basic 5000 500 0.01 10 g 2>&1 | tee output
        gprof ./nbody_basic gmon.out > profiling.txt
        ;;
    "red")
        echo "Run nbody_red in serial mode."
        gcc -pg -g -Wall -o nbody_red nbody_red.c -lm
        rm -rf output
        ./nbody_red 5000 500 0.01 10 g 2>&1 | tee output
        gprof ./nbody_red gmon.out > profiling.txt
        ;;
    "mpibasic")
        echo "Run nbody_basic in parallel mode."
        mpicc -g -Wall -o nbody_basic nbody_basic.c -lm
        rm -rf output
        ./nbody_basic 5000 500 0.01 10 g 2>&1 | tee output
        ;;
    "mpibasic")
        echo "Run nbody_red in parallel mode."
        gcc -g -Wall -o nbody_basic nbody_basic.c -lm
        rm -rf output
        ./nbody_basic 5000 500 0.01 10 g 2>&1 | tee output
        ;;
    "mpired")
        echo "n"
        ;;
    *)
        echo "选择了未知选项"
        ;;
esac

