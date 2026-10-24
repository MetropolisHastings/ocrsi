"""七层 21 个任务 Agent（依据论文附录 A.1）与五种引导风格（附录 A.2）。

论文只公开了 4 个 Agent 的完整提示词，其余 Agent 的主题与方向依据论文对各 Agent 的描述编写。
"""

AGENTS = {
    # Level I  Market Structure & Cycle
    "AgentMarketCycle": (1, "long-term cyclical transitions and phase shifts in price dynamics",
        "- trend/cycle phase from fast vs slow moving averages or their angle\n"
        "- position of price within multi-month ranges\n- turning-point detection"),
    "AgentVolatilityRegime": (1, "transitions between calm and turbulent volatility states",
        "- short vs long realized volatility ratio\n- volatility clustering and persistence\n"
        "- regime-shift detection in range-based volatility"),
    # Level II  Extreme Risk & Fragility
    "AgentTailRisk": (2, "downside sensitivity and tail-event exposure",
        "- downside semi-deviation, left-tail quantiles of returns\n"
        "- skewness / kurtosis of recent returns\n- frequency and size of large negative days"),
    "AgentCrashPredictor": (2, "early warning signals of collapses: volatility compression, liquidity depletion, fragility",
        "- volatility compression before expansion\n- shrinking volume while price stretches\n"
        "- distance from recent peak combined with weakening participation"),
    # Level III  Price–Volume Dynamics
    "AgentLiquidity": (3, "market depth and trading frictions through price impact and turnover variability",
        "- Amihud-style illiquidity (|return| / dollar volume)\n- price impact per unit of volume\n"
        "- volume shocks relative to history"),
    "AgentOrderImbalance": (3, "directional pressure from one-sided participation inferred from daily OHLCV",
        "- close location within the daily range weighted by volume\n"
        "- up-day vs down-day volume imbalance\n- accumulation/distribution style flows"),
    "AgentPriceVolumeCoherence": (3, "synchronization and divergence between price and volume changes",
        "- rolling correlation of returns and volume changes\n- price moves on rising vs falling volume\n"
        "- divergence between price trend and volume trend"),
    "AgentVolumeStructure": (3, "statistical shape and concentration of trading activity",
        "- dispersion, skewness or concentration of volume over a window\n"
        "- share of volume on the largest days\n- rhythm/clustering of participation"),
    # Level IV  Price–Volatility Behavior
    "AgentDailyTrend": (4, "directional persistence and multi-day momentum strength",
        "- risk-adjusted momentum\n- consistency of up days\n- trend efficiency (net move / path length)"),
    "AgentReversal": (4, "mean-reversion and short-term overreaction corrections",
        "- short-horizon return reversal\n- deviation from moving average scaled by volatility\n"
        "- overreaction after gaps or extreme days"),
    "AgentRangeVol": (4, "range-based volatility dynamics and compression–expansion cycles",
        "- Parkinson / Garman-Klass volatility\n- range contraction vs expansion\n- intraday range relative to history"),
    "AgentLagResponse": (4, "delayed price adjustments and lagged feedback between volatility, volume and returns",
        "- lagged correlations between volume and subsequent returns\n- autocorrelation structure of returns\n"
        "- response of returns to past volatility shocks"),
    "AgentVolAsymmetry": (4, "asymmetric volatility between upward and downward moves",
        "- upside vs downside volatility\n- asymmetry of gains and losses\n- leverage-effect style signals"),
    # Level V  Multi-Scale Complexity
    "AgentDrawdown": (5, "depth, duration and recovery geometry of cumulative losses",
        "- current drawdown from rolling peak\n- time since peak\n- speed of recovery from troughs"),
    "AgentFractal": (5, "multi-scale roughness and long-memory characteristics",
        "- variance ratio across horizons\n- Hurst-like rescaled range proxies\n- path roughness vs smoothness"),
    # Level VI  Stability & Regime-Gating
    "AgentRegimeGating": (6, "adaptive gates that switch signals on/off depending on volatility, trend or liquidity state",
        "- a base signal multiplied by a smooth gate (tanh/sigmoid) of a regime variable\n"
        "- different signals in high vs low volatility regimes"),
    "AgentStability": (6, "temporal consistency and persistence of returns or derived signals",
        "- stability of returns (mean/std over time)\n- smoothness of trends\n- persistence of a signal's sign"),
    # Level VII  Geometric & Fusion
    "AgentBarShape": (7, "candlestick geometry: body, shadows and symmetry as continuous descriptors",
        "- body to range ratio\n- upper vs lower shadow\n- rolling averages of bar-shape descriptors"),
    "AgentCreative": (7, "non-linear transformations, reparametrizations and soft gating of classic signals",
        "- tanh/sigmoid/log transforms of known signals\n- reparametrized classic indicators\n- soft thresholds"),
    "AgentComposite": (7, "fusion of several independent signals into coherent, orthogonal composites",
        "- product or ratio of two complementary signals\n- conditional combination of trend and volatility"),
    "AgentHerding": (7, "collective crowding behaviour and directional alignment within OHLCV dynamics",
        "- streaks of same-direction days with rising volume\n- crowding after strong runs\n- consensus intensity"),
}

# 论文 A.2：五种引导风格。论文用大模型改写提示词，这里用一句风格指令近似。
GUIDANCE = {
    "light": "Stay close to the classic, well-known formulations of this theme.",
    "moderate": "Use natural variations of standard formulations of this theme.",
    "creative": "Think like a researcher: propose alternative reasoning angles within this theme.",
    "divergent": "Explore related but different analytical perspectives that complement the obvious ones.",
    "concrete": "Translate each idea into a precise, simple mathematical or statistical formula.",
}
