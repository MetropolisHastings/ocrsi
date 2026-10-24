#!/bin/bash
# 给其余所有方法补跑“带组合回测”的在线实验（不影响已有结果，输出到 runs/frame/online_bt）。
# 用一个排队器控制并发：同时最多 P 个进程，跑完一个自动补一个。断线不影响（整体用 nohup 启动）。
# 用法：P=24 nohup bash run_bt_all.sh > logs/bt_all.log 2>&1 &
cd "$(dirname "$0")"
P=${P:-24}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_NUM_THREADS=2
METHODS=${METHODS:-"lasso elasticnet xgboost catboost doubleensemble gp mlp lstm gru alstm transformer gat adarnn tra rdagent"}
jobs=()
for m in $METHODS; do for s in 0 1 2 3 4; do for r in naive certified; do
  f="runs/frame/online_bt/online_${m}_${r}_s${s}.json"
  if [ -f "$f" ] && python -c "import json,sys;sys.exit(0 if len(json.load(open('$f'))['rounds'])==6 else 1)" 2>/dev/null; then
    continue                                    # 已经跑完的跳过
  fi
  jobs+=("$m $r $s")
done; done; done
echo "共 ${#jobs[@]} 个任务，并发 $P"
printf '%s\n' "${jobs[@]}" | xargs -P "$P" -L 1 bash -c '
  m=$0; r=$1; s=$2
  python online.py --method $m --rule $r --seed $s --bt --data ../qlib_bin \
    --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz \
    --out runs/frame/online_bt > logs/bt_${m}_${r}_s${s}.log 2>&1
  rc=$?
  echo "$(date +%H:%M) 完成 $m $r s$s（退出码 $rc）"'
echo "全部结束 $(date)"
