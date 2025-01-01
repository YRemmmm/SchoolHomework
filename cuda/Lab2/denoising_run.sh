rm -rf cpu*.pgm
rm -rf gpu*.pgm
rm -rf denoising
nvcc -o denoising denoising.cu
./denoising