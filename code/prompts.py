"""提示词模板（依据论文附录 C 精简改写）。"""

COMMON_RULES = """### Requirements
- The input `df` is the daily time series of ONE stock (already grouped by ticker),
  indexed by date, with columns: open, high, low, close, volume.
- Output: a pd.Series with the same index, named exactly as the function.
- Function name: factor_<logic>_<transformation>_<window>_<field>.
- Include a clear docstring explaining the logic and formula.
- Use only past and current data (no shift(-k), no future information).
- Nested loops and `while True` are strictly forbidden. Prefer vectorized pandas/numpy.
- Handle NaN / division by zero (add eps).
- Must be fast on 3,600 days x 800 stocks: do NOT use .rolling(...).apply(...), .apply with a
  Python lambda, or explicit Python loops over rows; use built-in vectorized pandas/numpy
  operations (rolling mean/std/max/min/rank/corr, ewm, shift, diff, etc.).
- At most 5 logical steps; one clear idea per factor.

### Pre-imported libraries
np (numpy), pd (pandas), stats (scipy.stats), math

### Output format
- Do NOT use markdown. Do NOT write anything outside the functions.
- Wrap each function as:
<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy["factor_xyz"]
<</function N>>
"""

AGENT_THEMES = {
    "AgentLiquidity": (
        "market liquidity, depth and trading friction",
        "- Amihud-style illiquidity (|return| / dollar volume)\n"
        "- price impact per unit of volume or turnover\n"
        "- volume shocks relative to recent history\n"
        "- interactions between liquidity and volatility or trend",
    ),
}

BASE_AGENT = """You are a senior quantitative factor engineer and an expert in {theme}.
Please generate {n} new and original quantitative factor functions that are distinct
from the existing ones. Each factor should be a complete Python function.

### Existing factors
{existing}

### Analysis of effective factors (Observation -> Cause -> Fix)
{effective}

### Analysis of ineffective factors (Observation -> Cause -> Fix)
{ineffective}

### Factor design guidance
Explore signals and techniques related to {theme}, including but not limited to:
{ideas}
Do not limit yourself to simple formulas; keep factors predictive, robust and
economically interpretable.

""" + COMMON_RULES

MUTATION = """You are an expert quantitative factor engineer specialized in factor mutation
and optimization. Generate an improved version of the parent factor below: keep its
core idea, but change ONE meaningful element (normalization, window, field, non-linear
transform) to make it more predictive and robust.

### Parent factor (train metrics: {metrics})
<<parent factor>>
{code}
<</parent factor>>

""" + COMMON_RULES

CROSSOVER = """You are an expert quantitative factor engineer specialized in factor evolution
and crossover design. Generate a new factor by intelligently combining the core insights
of the two parent factors. Introduce a meaningful (non-additive) interaction while keeping
a single clear financial intuition.

### Parent factor 1 (train metrics: {metrics1})
<<parent factor 1>>
{code1}
<</parent factor 1>>

### Parent factor 2 (train metrics: {metrics2})
<<parent factor 2>>
{code2}
<</parent factor 2>>

""" + COMMON_RULES


# ---------------- 多 Agent 质检（论文 C.2，精简） ----------------
JUDGE = """You are an expert quantitative researcher reviewing an alpha factor before it is
backtested. Do NOT guess its performance. Judge only its logic:
1. Logical consistency: does the code do what the docstring claims? Are signed and absolute
   quantities combined correctly (e.g. a price-impact slope must regress SIGNED return on
   SIGNED flow)?
2. Technical correctness: any division by quantities that are often near zero, explosive
   values, single-day inputs that make the signal pure noise, look-ahead (shift(-k)),
   meaningless constants?
3. Economic meaning: is there a clear, plausible financial intuition?

### Factor
{code}

Answer in exactly this format:
VERDICT: PASS or FAIL
ISSUES: <one short paragraph; write "none" if PASS>
"""

LOGIC_IMPROVE = """You are an expert quantitative factor engineer. A reviewer rejected the factor
below for these reasons:
{issues}

Rewrite the factor to fix every issue while keeping its core idea. Keep it simple.

### Factor
{code}

""" + COMMON_RULES

CODE_REPAIR = """You are an expert Python engineer for quantitative factors. The factor below
fails with this error:
{error}

Fix the code with minimal changes; keep the same function name and idea.

### Factor
{code}

""" + COMMON_RULES


# ---------------- 低成本批量版本 ----------------
COMPLEXITY = """### Hard complexity constraints (must follow)
- Simple factors are often the most powerful and stable: one clear idea per factor.
- Never exceed 5 logical steps; no decorative stacking such as zscore(zscore(x)).
- Do not mix unrelated themes.
"""

EVOLVE_BATCH = """You are an expert quantitative factor engineer specialized in factor evolution.
You are given two parent factors with their TRAIN metrics. Produce exactly three children:
<<function 1>>: MUTATION of Parent A - keep its core idea, change ONE meaningful element
               (normalization, window, field, or a non-linear transform).
<<function 2>>: CROSSOVER of Parent A and Parent B - combine their core insights through a
               meaningful, non-additive interaction with one clear financial intuition.
<<function 3>>: CROSSOVER + MUTATION - a crossover of A and B, then one further modification.
Each child must have a NEW, unique function name.

### Parent A (train metrics: {metrics_a})
{code_a}

### Parent B (train metrics: {metrics_b})
{code_b}

""" + COMPLEXITY + "\n" + COMMON_RULES

JUDGE_BATCH = """You are an expert quantitative researcher reviewing alpha factors before they are
accepted. Do NOT guess performance. For EACH factor judge only:
1. Logical consistency: does the code do what the docstring claims? Are signed and absolute
   quantities combined correctly?
2. Technical correctness: division by quantities often near zero, explosive values, single-day
   inputs that make the signal pure noise, look-ahead, meaningless constants.
3. Economic meaning: a clear, plausible financial intuition.

{factors}

Answer with exactly one line per factor, in order, in this format:
1: PASS - none
2: FAIL - <short reason>
"""
