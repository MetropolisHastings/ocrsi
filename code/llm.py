"""大模型接口。

两种模式：
- api：调用 OpenAI 兼容接口（vLLM 本地部署、DeepSeek、Qwen 等都支持）。
  需要环境变量 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL。
- manual：不调接口，从 responses/ 目录读取事先写好的回答（用于演示或调试）。
"""
import json
import os
import random
import re
import urllib.request
import urllib.error
import time
import threading
from pathlib import Path



_KIMI_LOCK = threading.Lock()
_KIMI_LAST_CALL = 0.0

def _kimi_rate_limit():
    global _KIMI_LAST_CALL
    if "moonshot.cn" not in os.environ.get("LLM_BASE_URL", ""):
        return
    with _KIMI_LOCK:
        wait = 21.0 - (time.monotonic() - _KIMI_LAST_CALL)
        if wait > 0:
            print(f"KIMI_RATE_LIMIT: waiting {wait:.1f}s", flush=True)
            time.sleep(wait)
        _KIMI_LAST_CALL = time.monotonic()

def chat(prompt: str, mode: str = "api", response_file: str | None = None,
         temperature: float | None = None) -> str:
    if mode == "manual":
        return Path(response_file).read_text(encoding="utf-8")
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    body = {
        "model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        # 论文：生成时温度在 0.7~1.2 之间随机
        # 有些服务商（如 Kimi）限制温度上限，用环境变量 LLM_MAX_TEMP 截断，默认不截断
        "temperature": 1,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}",
    })
    try:
        _kimi_rate_limit()
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"KIMI_HTTP_ERROR {e.code}: {detail}", flush=True)
        raise


def parse_functions(text: str) -> list[str]:
    """提取 <<function N>> ... <</function N>> 之间的代码。"""
    blocks = re.findall(r"<<\s*function\s*\d+\s*>>(.*?)<<\s*/\s*function\s*\d+\s*>>",
                        text, flags=re.S)
    if not blocks and "def factor_" in text:      # 模型没按格式包裹时，退而直接取函数
        start = text.index("def factor_")
        blocks = [text[start:].split("<<")[0].replace("```", "")]
    return [b.strip() for b in blocks]
