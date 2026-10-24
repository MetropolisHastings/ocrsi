#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_NUM_THREADS=2
METHODS=${METHODS:-"xgboost catboost doubleensemble gp rdagent gat tra transformer adarnn"}
SEEDS=${SEEDS:-"0 1 2 3 4"}; export LIMIT=${LIMIT:-46}
jobs=()
for s in $SEEDS; do for m in $METHODS; do for r in naive certified; do
  [ -f "runs/frame/online_bt/online_${m}_${r}_s${s}.json" ] && continue
  jobs+=("$m $r $s")
done; done; done
echo "共 ${#jobs[@]} 个任务"
printf '%s\n' "${jobs[@]}" | xargs -P 60 -L 1 bash -c '
  m=$0; r=$1; s=$2
  sleep $((RANDOM % 90))
  while [ $(pgrep -c -f "^python online.py") -ge $LIMIT ]; do sleep $((60 + RANDOM % 60)); done
  [ -f runs/frame/online_bt/online_${m}_${r}_s${s}.json ] && exit 0
  python online.py --method $m --rule $r --seed $s --bt --data ../qlib_bin \
    --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz \
    --out runs/frame/online_bt > logs/bt_${m}_${r}_s${s}.log 2>&1
  echo "$(date +%m-%d_%H:%M) 完成 $m $r s$s（退出码 $?）"'
echo "全部结束 $(date)"
