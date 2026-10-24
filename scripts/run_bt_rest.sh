#!/bin/bash
cd "$(dirname "$0")"
P=${P:-36}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_NUM_THREADS=2
METHODS="gat tra transformer adarnn alstm lstm gru mlp catboost doubleensemble xgboost gp rdagent"
jobs=()
for m in $METHODS; do for s in 0 1 2 3 4; do for r in naive certified; do
  [ -f "runs/frame/online_bt/online_${m}_${r}_s${s}.json" ] && continue
  jobs+=("$m $r $s")
done; done; done
echo "共 ${#jobs[@]} 个任务，并发 $P"
printf '%s\n' "${jobs[@]}" | xargs -P "$P" -L 1 bash -c '
  m=$0; r=$1; s=$2
  python online.py --method $m --rule $r --seed $s --bt --data ../qlib_bin \
    --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz \
    --out runs/frame/online_bt > logs/bt_${m}_${r}_s${s}.log 2>&1
  echo "$(date +%H:%M) 完成 $m $r s$s（退出码 $?）"'
echo "全部结束 $(date)"
