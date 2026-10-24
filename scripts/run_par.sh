# 每个“模型 × 种子 × 规则”单独一个进程并行跑，每个进程 3 线程
pkill -f "bash run_deep.sh"     # 只停掉排队用的外壳，正在跑的 python 会继续跑完
sleep 1
for m in transformer gat tra adarnn; do for s in 0 1 2; do for r in certified naive; do
  f=runs/frame/online/online_${m}_${r}_s${s}.json
  if [ -f $f ] && python -c "import json,sys;sys.exit(0 if len(json.load(open('$f'))['rounds'])==6 else 1)"; then continue; fi
  if pgrep -f "online.py --method $m --rule $r --seed $s " >/dev/null; then continue; fi
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nohup nice -n 10 python -u online.py --method $m --rule $r --seed $s \
    --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz --out runs/frame/online \
    >> runs/frame/online_${m}_s${s}_${r}.log 2>&1 &
  echo "启动 $m 种子$s $r"
done; done; done
