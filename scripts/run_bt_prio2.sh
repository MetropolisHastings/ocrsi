#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_NUM_THREADS=2
METHODS=${METHODS:-"xgboost catboost gp rdagent gat tra adarnn rf"}
export LIMIT=${LIMIT:-46}
busy(){ pgrep -f "^python online.py --method $1 --rule $2 --seed $3 " >/dev/null; }
jobs=()
for s in 0 1 2 3 4; do for m in $METHODS; do for r in naive certified; do
  [ -f "runs/frame/online_bt/online_${m}_${r}_s${s}.json" ] && continue
  busy $m $r $s && continue
  jobs+=("$m $r $s")
done; done; done
echo "共 ${#jobs[@]} 个任务"
for j in "${jobs[@]}"; do set -- $j
  while [ $(pgrep -c -f "^python online.py") -ge $LIMIT ]; do sleep 60; done
  [ -f runs/frame/online_bt/online_$1_$2_s$3.json ] && continue
  nohup python online.py --method $1 --rule $2 --seed $3 --bt --data ../qlib_bin \
    --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz \
    --out runs/frame/online_bt > logs/bt_$1_$2_s$3.log 2>&1 &
  echo "$(date +%m-%d_%H:%M) 启动 $1 $2 s$3"; sleep 20
done
echo "全部已启动 $(date)"
