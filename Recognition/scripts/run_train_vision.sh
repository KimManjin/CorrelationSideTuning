#!/usr/bin/env bash
if [ $1 -eq 0 ]; then
    gpus=0,1,2,3,4,5,6,7
    n_gpus=8
elif [ $1 -eq 1 ]; then
    gpus=0,1,2,3
    n_gpus=4
elif [ $1 -eq 2 ]; then
    gpus=4,5,6,7
    n_gpus=4
else
    echo "Invalid GPU setting"
    exit 1
fi

# num_threads=48
# num_workers=4

if [ -f $2 ]; then
  config=$2
else
  echo "need a config file"
  exit
fi

exp_name=$3

# Automatic port setting.
port=2335
while true; do
  # Check if the port is in use using netstat
  if netstat -tuln | awk '{print $4}' | grep -E "0.0.0.0:$port|:::$port" > /dev/null; then
    echo "Port $port is in use. Trying the next port."
    port=$((port + 1))
  else
    echo "Port $port is available."
    break
  fi
done

# MKL_NUM_THREADS=${num_threads} \
# NUMEXPR_NUM_THREADS=${num_threads} \
# OMP_NUM_THREADS=${num_threads} \
CUDA_VISIBLE_DEVICES=${gpus} \
python -m torch.distributed.run --master_port=${port} --nproc_per_node=${n_gpus} \
         train_vision.py  --config ${config} --exp_name ${exp_name}

