#!/bin/bash

DATASET=${1:-all}
DATA_TYPE=${2:-full}
SHOT=${3:-32}

python running.py \
  --dataset $DATASET \
  --data_type $DATA_TYPE \
  --shot $SHOT
