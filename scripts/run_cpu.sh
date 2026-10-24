m=$1
export OMP_NUM_THREADS=8
for s in 0 1 2 3 4; do for r in certified naive; do
  f=runs/frame/online/online_${m}_${r}_s${s}.json
  if [ -f $f ] && python -c "import json,sys;sys.exit(0 if len(json.load(open('$f'))['rounds'])==6 else 1)"; then continue; fi
  nice -n 10 python -u online.py --method $m --rule $r --seed $s --procs 8 \
    --full_cache ./csi300_panel_full.pkl --libcache runs/frame/lib_values.npz \
    --out runs/frame/online >> runs/frame/online_$m.log 2>&1
done; done
