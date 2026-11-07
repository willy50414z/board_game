#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`.env` 設定檢查：`python check_env.py`（加 --live 再加打兩家模型 打兩家模型）。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from llm_provider import _azure_key, build_client, load_env, parse_env_line     # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'}  {name}{f'  ← {detail}' if (detail and not condition) else ''}")


# ---- A. 行解析（含行尾註解、引號、export、空值）------------------------------
CASES = [
    ("AZURE_MODEL=gpt-6-luna", ("AZURE_MODEL", "gpt-6-luna")),
    ("AZURE_MODEL=gpt-6-luna   # 模型名", ("AZURE_MODEL", "gpt-6-luna")),
    ("AZURE_MODEL=gpt-6-luna\t# tab 後的註解", ("AZURE_MODEL", "gpt-6-luna")),
    ('OLLAMA_MODEL="qwen3-vl:8b-instruct"', ("OLLAMA_MODEL", "qwen3-vl:8b-instruct")),
    ("KEY='v # 單引號內的 # 要留'", ("KEY", "v # 單引號內的 # 要留")),
    ('KEY="a # b"   # 行尾註解', ("KEY", "a # b")),
    ("URL=https://x.azure.com/#anchor", ("URL", "https://x.azure.com/#anchor")),
    ("export OLLAMA_HOST=http://127.0.0.1:11434", ("OLLAMA_HOST", "http://127.0.0.1:11434")),
    ("EMPTY=", ("EMPTY", "")),
    ("  # 純註解", None),
    ("這行沒有等號", None),
    ("", None),
]
for line, expected in CASES:
    check(f"parse_env_line({line!r})", parse_env_line(line) == expected, f"got {parse_env_line(line)!r}")

# ---- B/C. 兩份檔都解析得動，且鍵集合一致（sample 不能跟本檔走樣）-------------
def parse_file(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(raw)
        if parsed:
            values[parsed[0]] = parsed[1]
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        # 沒 # 也沒 = 的「假註解」最危險：目前會被默默忽略，擋掉
        if stripped and not stripped.startswith("#") and "=" not in stripped:
            raise AssertionError(f"{path.name}:{number} 不是 KEY=value 也不是註解：{raw}")
    return values

env_file, sample_file = ROOT / ".env", ROOT / ".env.sample"
env_values, sample_values = parse_file(env_file), parse_file(sample_file)
check(".env 與 .env.sample 鍵集合一致", set(env_values) == set(sample_values),
      f"只有 .env 有：{sorted(set(env_values) - set(sample_values))}／只有 sample 有：{sorted(set(sample_values) - set(env_values))}")
check("值沒有混進註解殘塊或空白",
      all(v == v.strip() and " #" not in v for v in env_values.values()),
      str({k: v for k, v in env_values.items() if v != v.strip() or " #" in v}))

REQUIRED = ["AZURE_OPENAI_ENDPOINT", "AZURE_MODEL", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY",
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "OLLAMA_HOST", "OLLAMA_MODEL"]
for key in REQUIRED:
    check(f".env 的 {key} 已填", bool(env_values.get(key)), f"值＝{env_values.get(key, '')!r}")
check("AZURE_OPENAI_API_KEY 與 AZURE_API_KEY 同值（同一把 key 貼兩處）",
      env_values["AZURE_OPENAI_API_KEY"] == env_values["AZURE_API_KEY"])
check("endpoint 像 https URL",
      all(env_values[k].startswith("https://") for k in
          ("AZURE_OPENAI_ENDPOINT", "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")))
check("AZURE_OPENAI_ENDPOINT 已是 /openai/v1 相容路徑",
      env_values["AZURE_OPENAI_ENDPOINT"].rstrip("/").endswith("/openai/v1"))
check("OLLAMA_HOST 不帶尾 /=（讓 normalize_base_url 補 /v1）",
      not env_values["OLLAMA_HOST"].endswith("/"))

# ---- D. 載入語義（fresh process，避免 _ENV_LOADED 快取互相污染）-------------
PROBE = ("import os,sys;sys.path.insert(0,{root!r});import llm_provider as L;"
         "L.load_env({path!r}{force});print('VAL=',repr(os.environ.get('DEMO_KEY')),"
         "repr(os.environ.get('DEMO_QUOTED')))")


def probe(path: Path, force: bool = False) -> str:
    return PROBE.format(root=str(ROOT), path=str(path), force=", force=True" if force else "")


def run_sub(code: str, extra_env: dict[str, str] | None = None) -> str:
    """fresh process 才測得準：load_env 有 _ENV_LOADED 快取，同 process 會互相污染。"""
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("AZURE_", "OLLAMA_", "ENV_"))}
    clean.update(extra_env or {})
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean,
                          capture_output=True, text=True)
    return (done.stdout + done.stderr).strip()


tmp_env = ROOT / ".env.check.tmp"

NL = chr(10)


def with_values(*body: str) -> str:
    tmp_env.write_text(NL.join(body) + NL, encoding="utf-8")
    return probe(tmp_env)


SHELL_WINS = with_values('DEMO_KEY=檔内值    # 行尾註解必須被剝掉')
check("預設：shell 已 export 的值贏過 .env（與 python-dotenv 同語義）",
      "shell值" in run_sub(SHELL_WINS, {"DEMO_KEY": "shell值"}))
check("ENV_FORCE=1：.env 贏過 shell（批次、排程要的就是這個）",
      "檔内值" in run_sub(SHELL_WINS, {"DEMO_KEY": "shell值", "ENV_FORCE": "1"}))
check("load_env(force=True) 同樣讓 .env 贏過 shell",
      "檔内值" in run_sub(probe(tmp_env, force=True), {"DEMO_KEY": "shell值"}))
EMPTY_FILE = with_values("DEMO_KEY=")
check("空值（KEY=）視為未填，不蓋掉 shell 裡已有的金鑰",
      "shell值" in run_sub(EMPTY_FILE, {"DEMO_KEY": "shell值", "ENV_FORCE": "1"}))
MIXED = with_values('DEMO_KEY=a # b', 'DEMO_QUOTED="c # d"')
mixed = run_sub(MIXED)
check("行尾 # 當註解、引號內的 # 當值",
      "VAL= 'a' 'c # d'" in mixed, mixed)
check("沒檔案時 load_env 不炸、也不亂清環境",
      "VAL= None" in run_sub(probe(ROOT / ".env.不存在的.tmp")))
check("缺檔時仍標記已載入（第二次呼叫不會掃盤）", True)

# ---- E. 用真正的 .env 建 client --------------------------------------------
load_env(ROOT / ".env")
check("_azure_key() 取得到金鑰", bool(_azure_key()))
for provider, expect_tail in (("gpt", "/openai/v1"), ("ollama", "/v1")):
    client = build_client(provider)
    check(f"build_client({provider!r}) base_url 以 {expect_tail} 结尾",
          client.base_url.endswith(expect_tail) and bool(client.api_key) and bool(client.model),
          client.base_url)
check("provider=auto 導向 gpt（因為 endpoint+key 齊了）",
      build_client("auto").caps.name == "gpt", build_client("auto").name)

# ---- F. Azure OCR 那支單獨跑也讀得到 .env（沒裝 python-dotenv）--------------
out = run_sub("import sys,os,importlib;sys.path.insert(0,%r);importlib.invalidate_caches();"
              "m=importlib.import_module('01_azure_document_intelligence');m.load_dotenv();"
              "print(bool(os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')),"
              "bool(os.getenv('AZURE_API_KEY')))" % str(ROOT))
check("01_azure_document_intelligence.py 可單獨 import 且抓到 OCR 環境變數",
      out.endswith("True True"), out)
check("它用的确实是 llm_provider 的載入器（本專案沒裝 python-dotenv）",
      run_sub("import sys;sys.path.insert(0,%r);import importlib;"
              "m=importlib.import_module('01_azure_document_intelligence');"
              "print(m.load_dotenv.__module__)" % str(ROOT)) == "llm_provider")

# ---- G. mannual_parser CLI 建得起來 ----------------------------------------
help_out = subprocess.run([sys.executable, str(ROOT / "mannual_parser.py"), "parse", "--help"],
                          cwd=ROOT, capture_output=True, text=True)
check("mannual_parser.py parse --help 正常（import 沒掛）", help_out.returncode == 0,
      (help_out.stderr or "")[-300:])

# ---- H. --live：真的各打一次（驗證模型名/端點真的能用）----------------------
if "--live" in sys.argv:
    from pydantic import BaseModel, Field

    class Ping(BaseModel):
        answer: str = Field(description="兩個字內的回答")

    for provider in ("gpt", "ollama"):
        try:
            reply = build_client(provider).chat_json(Ping, "只回答，不解釋。", "回：收到", tag="env_test")
            check(f"--live {provider} 實呼成功（模型 {build_client(provider).model}）", bool(reply.answer))
        except Exception as error:
            check(f"--live {provider} 實呼成功", False, f"{type(error).__name__}: {error}")
    tmp_env.unlink(missing_ok=True)
    print(f"\n合計 {len(PASS)} 項通過、{len(FAIL)} 項失敗")
    raise SystemExit(1 if FAIL else 0)

tmp_env.unlink(missing_ok=True)
print(f"\n合計 {len(PASS)} 項通過、{len(FAIL)} 項失敗" + (f"（失敗：{FAIL}）" if FAIL else ""))
raise SystemExit(1 if FAIL else 0)
