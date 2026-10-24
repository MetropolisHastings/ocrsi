# OC-RSI: Online Certified Recursive Self-Improvement for Factor Mining

Anonymous code and logs for the WWW 2027 submission
"Certified Recursive Self-Improvement Delivers Verifiable Gains in LLM-Driven Factor Mining".

## What is here

| Path | Contents |
|---|---|
| `code/online.py` | The OC-RSI protocol: selection on [t-2, t-1], block-bootstrap confirmation on year t, deployment scored on t+1. `--rule naive` is the accept-if-better control. `--bt` adds the Top-50 portfolio backtest. |
| `code/rsiframe.py`, `code/orchestra.py` | The harness: the improvement-operator interface, the screening step and the confirmation test. |
| `code/deep.py`, `code/gp.py`, `code/agents.py`, `code/llm.py`, ... | Improvement operators for the 22 methods (linear, tree, deep, genetic programming, LLM agents). |
| `code/backtest.py` | TopK-Dropout backtest: top 50, at most 5 swaps a day, 0.05% buy and 0.15% sell costs, no trades at limit-up or limit-down. |
| `code/cot_online_bt.py`, `code/alphaagent_online_bt.py` | Direct prompting and themed prompting, run in a separate implementation of the same protocol. |
| `code/online_report.py`, `code/online_bt_report2.py`, `code/ic_table.py` | Scripts that produce Table 1, Table 2 and the appendix tables from the logs. |
| `scripts/` | The run queues used for the paper. |
| `logs.tar.gz` → `logs/online/` | Per-round logs behind Table 1 (22 methods x 2 rules x 5 runs x 6 rounds). |
| `logs.tar.gz` → `logs/online_bt/` | Per-round logs with daily portfolio returns behind Table 2 (21 methods). |
| `logs.tar.gz` → `logs/online_null0/` | The shuffled-factor placebo. |
| `logs.tar.gz` → `logs/direct_themed_original/` | Per-round records of direct and themed prompting behind Table 1. |

Every log file is one run (`online_{method}_{rule}_s{seed}.json`). Each round records
the candidates, the screened batch, the confirmation result (gain, lower bound, pass),
the accept/reject decision, the deployment-year RankIC, RankICIR and neutralized RankIC
before and after the update, and, in `online_bt/`, the daily excess return of the
deployed and the unchanged starting state.

Unpack the logs with `tar xzf logs.tar.gz`.

## Reproducing

Data: CSI300 constituents in Qlib binary format (daily OHLCV, 2008-2026). The market
data are not redistributed; any Qlib CN dataset with the same fields works.

    pip install -r requirements.txt
    cd code
    python online.py --method lightgbm --rule certified --seed 0 --bt \
        --data <qlib_dir> --full_cache ./csi300_panel_full.pkl \
        --libcache runs/frame/lib_values.npz --out runs/frame/online_bt
    python online_bt_report2.py runs/frame/online_bt <qlib_dir>

LLM methods read `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` from the environment
(the paper uses the `deepseek-chat` endpoint). No key is stored in this repository.

Main settings: alpha = 0.1, R = 6 rounds (decision years 2020-2025), per-round level
alpha/R, moving block bootstrap with 20-day blocks and B = 2000 resamples, 10-day
open-to-open forward return label.
