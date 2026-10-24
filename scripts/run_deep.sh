#!/bin/bash
# 在 zibs 上并行跑 4 个深度模型（每个模型一个进程，依次跑各种子和两种规则）。
# 用法：bash run_deep.sh "transformer gat tra adarnn" "0 1 2" 12
#   第 1 个参数：模型；第 2 个：种子；第 3 个：每个进程的线程数（4 个进程 × 12 = 48 线程）
MODELS=${1:-"transformer gat tra adarnn"}; SEEDS=${2:-"0 1 2"}; TH=${3:-12}
mkdir -p runs/frame/online
for m in $MODELS; do
  (
    export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH
    for seed in $SEEDS; do for rule in certified naive; do
      f=runs/frame/online/online_${m}_${rule}_s${seed}.json
      if [ -f $f ] && python -c "import json,sys;sys.exit(0 if len(json.load(open('$f'))['rounds'])==6 else 1)"; then continue; fi
      nice -n 10 python -u online.py --method $m --rule $rule --seed $seed \
        --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz \
        --out runs/frame/online >> runs/frame/online_${m}.log 2>&1
    done; done
    echo "$m DONE" >> runs/frame/deep_done.txt
  ) &
done
wait
