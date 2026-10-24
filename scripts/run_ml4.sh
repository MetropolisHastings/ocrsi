#!/bin/bash
# 同学那 4 个模型按在线协议重跑：4 模型 × 3 种子 × 2 规则 = 24 个进程，每个 3 线程。
# 数据缓存路径自动从 run_par.sh 里读取（和之前 7 个模型完全一致）。
cd "$(dirname "$0")"
FULL=$(grep -o -- '--full_cache [^ ]*' run_par.sh | head -1 | awk '{print $2}')
LIBC=$(grep -o -- '--libcache [^ ]*' run_par.sh | head -1 | awk '{print $2}')
OUT=$(grep -o -- '--out [^ ]*' run_par.sh | head -1 | awk '{print $2}')
DATA=$(grep -o -- '--data [^ ]*' run_par.sh | head -1 | awk '{print $2}')
[ -z "$FULL" ] || [ -z "$LIBC" ] || [ -z "$OUT" ] && { echo "没能从 run_par.sh 读到缓存路径，请手动填写"; exit 1; }
echo "full_cache=$FULL libcache=$LIBC out=$OUT data=${DATA:-默认}"
python -c "import lightgbm, xgboost, catboost" 2>/dev/null || pip install -q lightgbm xgboost catboost
mkdir -p "$OUT" logs
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
MODELS=${MODELS:-"lasso elasticnet lightgbm xgboost"}
for m in $MODELS; do
  for s in 0 1 2; do
    for r in naive certified; do
      nohup python online.py --method $m --rule $r --seed $s \
        --full_cache "$FULL" --libcache "$LIBC" --out "$OUT" ${DATA:+--data "$DATA"} \
        > logs/online_${m}_${r}_s${s}.log 2>&1 &
    done
  done
done
echo "已启动：$MODELS（每个 3 种子 × 2 规则）"
