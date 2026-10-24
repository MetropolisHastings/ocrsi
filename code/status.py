import json, glob, subprocess
run = subprocess.run("ps aux | grep 'online.py' | grep -v grep", shell=True, capture_output=True, text=True).stdout
total = {"mlp": 10, "ridge": 10, "gp": 10, "transformer": 6, "gat": 6, "tra": 6, "adarnn": 6}
print(f"正在运行的进程：{run.count('online.py')} 个\n")
for m, n in total.items():
    done, doing = 0, []
    for f in glob.glob(f"runs/frame/online/online_{m}_*.json"):
        k = len(json.load(open(f))["rounds"])
        if k == 6:
            done += 1
        else:
            doing.append(f"{f.split('_', 2)[2][:-5]} 第{k}/6轮")
    print(f"{m:12s} 完成 {done}/{n} 组" + (f"；进行中：{', '.join(doing)}" if doing else ""))
