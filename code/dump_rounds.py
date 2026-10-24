import json, sys
from pathlib import Path
import pandas as pd
NAMES = {"orch":"LLM-Orch","reflect":"Reflexion","rdagent":"RD-Agent(Q)","gat":"GATs","rf":"RandomForest","dens":"DoubleEnsemble"}
def flat(p, o, out):
    for k, v in o.items():
        if isinstance(v, dict):
            flat(f"{p}{k}_", v, out)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[p+k] = v
rows = []
d = Path(sys.argv[1])
for f in sorted(d.glob("online_*_s*.json")):
    j = json.loads(f.read_text())
    if len(j["rounds"]) < 6:
        continue
    for r in j["rounds"]:
        row = {"method": NAMES.get(j["method"], j["method"]), "rule": j["rule"], "seed": j["seed"]}
        flat("", r, row)
        row["n_kept"] = len(r.get("kept") or [])
        rows.append(row)
t = pd.DataFrame(rows)
t.to_csv(d/"rounds_all.csv", index=False, encoding="utf-8-sig")
print("行数", t.shape)
print("列名：", ", ".join(t.columns))
print(t.head(3).to_string())
