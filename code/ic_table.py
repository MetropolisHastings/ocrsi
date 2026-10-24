import collections, glob, json, sys
import numpy as np
d = sys.argv[1]; KEY = sys.argv[2]; KEY = sys.argv[2]
dep = collections.defaultdict(list); start = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{d}/online_*_s*.json")):
    j = json.load(open(f))
    if len(j["rounds"]) < 6: continue
    m, r = j["method"], j["rule"]; vals, untouched = [], True
    for row in j["rounds"]:
        y = row["deploy_year"]
        if untouched: start[m][y] = row["deploy_before"][KEY]
        vals.append((row["deploy_after_candidate"] if row["accepted"] else row["deploy_before"])[KEY])
        if row["accepted"]: untouched = False
    dep[(m, r)].append(float(np.mean(vals)))
years = list(range(2021, 2027)); rows = []
print(f"{'方法':16s}{'起点':>9s}{'naive':>9s}{'OC-RSI':>9s}   缺失年份")
for m in sorted(start):
    miss = [y for y in years if y not in start[m]]
    s0 = np.mean([start[m][y] for y in years if y in start[m]])
    n = np.mean(dep[(m, "naive")]); c = np.mean(dep[(m, "certified")])
    print(f"{m:16s}{s0:9.4f}{n:9.4f}{c:9.4f}   {miss or ''}"); rows.append((m, s0, n, c, miss))
json.dump(rows, open(f"{d}/ic_table.json", "w"))
