#/bin/bash

set -x

input_file=$1
output_file=${input_file%.*}.out

mpic++ -g $input_file -o $output_file -I ../Libraries/include -L ../Libraries/lib -lcudart -lmp -lgdsync -lgdrapi
