#!/bin/bash


window_sizes=(1 3 5 7)
for window_size in ${window_sizes[@]}
do
  echo ${window_size}
  #sbatch -p a --gres=gpu:1 --mem 60G -t 12:00:00 --export window_size=${window_size},PATH=$PATH -o run_wz${window_size}.out run.sh
  #sbatch -p big --mem 60G -t 12:00:00 --export window_size=${window_size},PATH=$PATH -o run_baseline_wz${window_size}.out run_baseline.sh
  sbatch -p a --gres=gpu:1 --mem 60G -t 12:00:00 --export window_size=${window_size},PATH=$PATH -o run_wz${window_size}.out run_baseline.sh
done
