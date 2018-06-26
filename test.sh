#!/bin/bash

root=${1%/*}
# default port is set to 6007
port=${2:-6007}

models=$(ls -l $root | awk '/^d/ {print $NF}')
path=()
for model in $models
do
  repeats=$(ls -l "$root/$model" | awk '/^d/ {print $NF}')
  for repeat in $repeats
  do
    # name:path_of_log
    path=(${path[*]} "$root/$model/$repeat/test_log:$root/$model/$repeat/test_log")
  done
done
# array to string delimited by ','
path_concat=$(IFS=','; echo "${path[*]}")

$(tensorboard --port=$port --logdir=$path_concat)

