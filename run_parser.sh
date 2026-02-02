#!/bin/bash

# ====== 可配置参数 ======
DATASET=all        # 单个数据集如 Apache / BGL，或 all
DATA_TYPE=full     # full / small（如果你有）
SHOT=32            # 0 / 16 / 32 / 64 / 128

# ====== 运行 ======
python running.py \
  --dataset $DATASET \
  --data_type $DATA_TYPE \
  --shot $SHOT \