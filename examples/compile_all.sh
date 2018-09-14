#!/bin/bash

set -x

CUDA_PATH="/cm/extra/apps/CUDA.linux86-64/9.2.88.1_396.26"
input_file=$1
object_file=${input_file%.*}.o
output_file=${input_file%.*}.out

nvcc -c -g $input_file -o $object_file -I ../Libraries/include
mpic++ -g $object_file -o $output_file -I ../Libraries/include -I $CUDA_PATH/include -L ../Libraries/lib -L $CUDA_PATH/lib64 -lcuda -lcudart -lmp -lgdsync -lgdrapi -lnvToolsExt
