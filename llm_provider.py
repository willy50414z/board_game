#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider 接觸層：**一條 OpenAI 相容路徑** ＋ **pydantic 當 schema 唯一來源**。

之前 Ollama 走 SDK（format=schema、images=[路徑]）、Azure 走 REST（response_format、
image_url），參數與圖片形狀各一套，if provider 散在呼叫端。這裡收斂成：

  1. 只打 `<base_url>/chat/completions`：Azure 的 `/openai/v1`、Ollama 的 `/v1`
     都是 OpenAI 相容端點，payload 形狀一致，provider 差異只剩「能力表」。
  2. 結構化輸出用 pydantic model：`chat_json(SomeReply, ...)` 送出去的是
     `SomeReply.model_json_schema()`（濾成各家收的關鍵字），收回來的一定是
     通過驗證的 `SomeReply` 實體。欄位描述只寫一处。
  3. 各家對參數的怪癖（`max_completion_tokens` vs `max_tokens`、不收 temperature、
     不認得新增欄位）由 `_heal()` 照錯誤訊息就地修正重打，不進 schema、不進業務碼。

約束：回覆模型（Reply）的欄位**一律必填、不給 default**，這樣才符合 strict mode
「每個 object 的每個 property 都要 required」的規定；可选值請用 `str` 留空字串或
`Literal[...]`，不要寫 `Optional[...]`。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

PROVIDER_GPT = "gpt"
PROVIDER_OLLAMA = "ollama"
PROVIDER_ALIASES = {
    "gpt": PROVIDER_GPT, "azure": PROVIDER_GPT, "openai": PROVIDER_GPT,
    "ollama": PROVIDER_OLLAMA,
}
DEFAULT_GPT_MODEL = "gpt-6-luna"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


# ---------------------------------------------------------------------------
# 能力表：provider 之間的差異全部登记在此，業務碼不再 if provider
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Capability:
    """某個 OpenAI 相容端點「哪裡不一樣」的單一紀錄處。"""

    name: str
    default_model: str
    default_base_url: str
    det_long_side: int                  # 送進模型的偵測圖長邊
    needs_key: bool = False
    strict_schema: bool = True          # 支援 response_format=json_schema(strict)
    json_object: bool = True            # 降階時可用 json_object 模式
    image_detail: bool = False          # image_url 要不要附 detail 欄位
    temperature: float | None = None    # None＝不送，用服務端預設
    token_param: str = "max_tokens"     # 長度欄位叫什麼
    extra_payload: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 8192
    timeout: int = 600
    note: str = ""


CAPABILITIES: dict[str, Capability] = {
    PROVIDER_GPT: Capability(
        name=PROVIDER_GPT,
        default_model=DEFAULT_GPT_MODEL,
        default_base_url="",                       # 一定從 AZURE_OPENAI_ENDPOINT 來
        det_long_side=2048,
        needs_key=True,
        image_detail=True,
        temperature=None,                          # gpt-6-luna 只收預設值
        token_param="max_completion_tokens",
        note="Azure AI Foundry 的 gpt-6-luna：/openai/v1 相容端點",
    ),
    PROVIDER_OLLAMA: Capability(
        name=PROVIDER_OLLAMA,
        default_model=DEFAULT_OLLAMA_MODEL,
        default_base_url=DEFAULT_OLLAMA_BASE_URL,
        det_long_side=1568,
        temperature=0.1,
        token_param="max_tokens",
        # OpenAI 相容端點沒有 num_ctx；Ollama 放行外層 options（不認得的話 _heal 會丟掉）
        extra_payload={"options": {"num_ctx": 16384}},
        note="本機 Ollama 的 /v1 OpenAI 相容端點",
    ),
}


_ENV_LOADED = False
_TRUE = ("1", "true", "yes", "y", "on")


def parse_env_line(line: str) -> tuple[str, str] | None:
    """`KEY=value   # 註解` → ("KEY", "value")；註解行／空白行／沒有 `=` 的爛行 → None。

    行尾註解只認「空白後的 #」，值裡的 #（URL 錨點）才會留下來；
    成對引號剝掉（`KEY="a # b"` 值是 `a # b`）。舊版把整行連註解一起當值塞進
    os.environ，害得 `.env.sample` 那種寫法一填就得到 `gpt-6-luna   # 模型名`。
    """
    text = line.strip()
    if not text or text.startswith("#") or "=" not in text:
        return None
    key, _, value = text.partition("=")
    key = key.strip()
    if key.startswith("export "):       # 有人會直接把 shell 的 export 行貼進來
        key = key[len("export "):].strip()
    value = value.strip()
    quote = value[:1]
    if quote in ("'", '"'):
        closing = value.find(quote, 1)
        if closing > 0:           # 引號內原樣保留（裡面的 # 不是註解），尾端註解整段丟掉
            value = value[1:closing].strip()
            return (key, value) if key else None
        value = value.strip(quote)        # 只有一半引號：退回舊行為，別把引號當值
    comment = re.search(r"\s#", value)
    if comment:
        value = value[:comment.start()].strip()
    return (key, value) if key else None


def load_env(env_path: Path | None = None, force: bool = False) -> None:
    """很小的 .env 載入器（不拉 python-dotenv 依賴）；`build_client()` 會自動呼叫。

    預設 **shell 贏過檔案**（`os.environ.setdefault`，與 python-dotenv 同語義），
    這樣 `export OLLAMA_HOST=...` 那種一次性覆寫才有效；要讓 `.env` 說了算
    （批次、排程、CI）就在 shell 設 `ENV_FORCE=1`，或直接 `load_env(force=True)`。
    空值（`KEY=`）代表「這欄沒填」，一律不覆蓋既有值。
    """
    global _ENV_LOADED
    force = force or str(os.getenv("ENV_FORCE", "")).strip().lower() in _TRUE
    if _ENV_LOADED and not force:
        return
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    _ENV_LOADED = True          # 沒檔案也標起來，避免每次 build_client 都掃 disk
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if not value:
            os.environ.setdefault(key, "")
        elif force:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def resolve_provider(name: str) -> str:
    key = str(name or "auto").strip().lower()
    if key not in PROVIDER_ALIASES:
        raise ValueError(f"不認識的 provider: {name}（可選 auto/ollama/gpt/openai/azure）")
    return PROVIDER_ALIASES[key]


def auto_provider() -> str:
    """沒指定就跟著 .env 走：有 Azure 金鑰就優先雲端 GPT。"""
    has_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT")) and bool(_azure_key())
    return PROVIDER_GPT if has_azure else PROVIDER_OLLAMA


def _azure_key() -> str:
    return os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or ""


def normalize_base_url(raw: str, caps: Capability) -> str:
    """補成可直接 `+/chat/completions` 的基底（兩家都吃 OpenAI 相容路徑）。"""
    url = (raw or caps.default_base_url or "").rstrip("/")
    if not url:
        raise RuntimeError(f"provider {caps.name} 沒有 base_url（請設 env 或帶 --host）")
    if caps.name == PROVIDER_OLLAMA and not re.search(r"/v1$", url):
        url = url + "/v1"          # 給人 Ollama 根位址也能動
    if caps.name == PROVIDER_GPT and not re.search(r"/v1(/)?$|/openai", url):
        url = url + "/openai/v1"   # 只給 ...openai.azure.com 時補上相容路徑
    return url


# ---------------------------------------------------------------------------
# pydantic model → 各家收得的 JSON Schema
# ---------------------------------------------------------------------------

# 白名單式濾鏡：實測 GPT 的 structured output 只吃這些關鍵字，多一個就 400
# （minItems/maxItems、title、format、$schema、examples 都会被唸）
_ALLOWED_SCHEMA_KEYS = (
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "const", "description", "anyOf", "oneOf", "allOf", "not", "nullable", "pattern",
)


def _resolve_ref(node: dict[str, Any], defs: dict[str, Any], seen: tuple[str, ...] = ()) -> Any:
    ref = str(node.get("$ref") or "")
    match = re.match(r"^#/\$defs/(.+)$", ref)
    if not match:
        return node
    name = match.group(1)
    if name in seen:                      # 遞迴模型：停在此處，避免無限展開
        return {}
    return _expand(defs.get(name, {}), defs, seen + (name,))


def _expand(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_expand(item, defs, seen) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return _resolve_ref(node, defs, seen)
    return {key: _expand(value, defs, seen) for key, value in node.items()}


def _sanitize(node: Any, strict: bool) -> Any:
    """濾掉服務端不認得的關鍵字；strict 時補上 strict mode 硬要求的兩件事。"""
    if isinstance(node, list):
        return [_sanitize(item, strict) for item in node]
    if not isinstance(node, dict):
        return node

    result: dict[str, Any] = {}
    for key, value in node.items():
        # properties 的 key 是【欄位名】不是關鍵字，濾不得（濾了會被唸 required 多鍵）
        if key == "properties" and isinstance(value, dict):
            result[key] = {name: _sanitize(sub, strict) for name, sub in value.items()}
        elif key == "required":
            result[key] = list(value)
        elif key in _ALLOWED_SCHEMA_KEYS:
            result[key] = _sanitize(value, strict)

    if strict and ("properties" in result or result.get("type") == "object"):
        result["additionalProperties"] = False
        if result.get("properties"):
            # strict mode 要求 object 的所有 property 都必填，所以 Reply 欄位不給 default
            result["required"] = list(result["properties"])
    return result


def openai_schema(model: type[BaseModel], strict: bool = True) -> dict[str, Any]:
    """pydantic model → 可直接塞進 response_format 的 JSON Schema（$ref 已就地展開）。"""
    raw = model.model_json_schema(ref_template="#/$defs/{model}")
    expanded = _expand(raw, raw.get("$defs") or {}, ())
    return _sanitize(expanded, strict)


def format_instructions(model: type[BaseModel]) -> str:
    """降階（寬鬆 JSON 模式）時把 schema 寫進 prompt，不然模型只看過 schema 一次。"""
    schema = json.dumps(openai_schema(model, strict=True), ensure_ascii=False,
                        separators=(",", ":"))
    return f"\n\n只輸出一個符合此 JSON Schema 的 JSON 物件（不要任何解釋、不要 markdown）：\n{schema}"


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    blob = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{blob}"


# 少了就一定跑不起來的欄位：自愈時永不當成「多餘參數」丟掉
_CRITICAL_KEYS = ("model", "messages")
# 錯誤訊息裡點名欄位慣用引號包住（'temperature'、"response_format"）
_QUOTED_KEY_RE = re.compile(
    "[" + re.escape("\"'`") + r"]([A-Za-z_][A-Za-z0-9_.]*)["
    + re.escape("\"'`") + "]")


class ProviderError(RuntimeError):
    """呼叫端只需要知道「這類請求失敗」，細節在訊息與 _debug/llm_raw 裡。"""


# ---------------------------------------------------------------------------
# 唯一那條 request 路徑
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    caps: Capability
    model: str
    base_url: str
    api_key: str = ""
    raw_dir: Path | None = None
    temperature: float | None = None       # None＝用能力表的值
    max_tokens: int = 0                    # 0＝用能力表的值
    timeout: int = 0                       # 0＝用能力表的值
    verbose: bool = True
    last_usage: dict[str, Any] = field(default_factory=dict, repr=False)
    total_usage: dict[str, Any] = field(default_factory=dict, repr=False)
    calls: int = field(default=0, repr=False)
    _session: Any = field(default=None, repr=False)

    # -- 基本欄位 -----------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{self.caps.name}:{self.model}"

    @property
    def det_long_side(self) -> int:
        """送進模型的偵測圖長邊（provider 能力，不是業務參數）。"""
        return self.caps.det_long_side

    def _temperature(self) -> float | None:
        return self.caps.temperature if self.temperature is None else self.temperature

    def _max_tokens(self) -> int:
        return self.max_tokens or self.caps.max_output_tokens

    def _http(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def dump_raw(self, tag: str, content: str) -> None:
        """每次呼叫都留原始回應（含失敗的），事後才知道模型到底給了什麼。"""
        if self.raw_dir is None or not content:
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
        (self.raw_dir / f"{safe}.txt").write_text(content, encoding="utf-8")

    # -- payload ------------------------------------------------------------
    def _user_content(self, prompt: str, images: list[Path]) -> Any:
        if not images:
            return prompt
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in images:
            entry: dict[str, Any] = {"type": "image_url",
                                     "image_url": {"url": data_url(Path(path))}}
            if self.caps.image_detail:
                entry["detail"] = "high"
            content.append(entry)
        return content

    def _payload(self, system: str, prompt: str, images: list[Path],
                 schema: type[BaseModel] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            self.caps.token_param: self._max_tokens(),
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": self._user_content(prompt, images)}],
        }
        payload.update(self.caps.extra_payload)
        temperature = self._temperature()
        if temperature is not None:
            payload["temperature"] = temperature
        if schema is not None and self.caps.strict_schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "reply", "strict": True, "schema": openai_schema(schema, strict=True)}}
        elif self.caps.json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _heal(self, payload: dict[str, Any], detail: str,
              tried: tuple[str, ...]) -> tuple[dict[str, Any], str] | None:
        """照錯誤訊息就地修正：被唸到哪個參數就修哪個，然後重打一次。

        三條規則由外而內：長度欄位改名 → 訊息點名的欄位 → payload 現存的加料欄位。
        """
        # 1) 長度欄位的名字兩家不同（gpt-6-luna 只收 max_completion_tokens）
        rename = {"max_completion_tokens": "max_tokens", "max_tokens": "max_completion_tokens"}
        for key, replacement in rename.items():
            if key in payload and key not in tried and key in detail:
                fixed = dict(payload)
                fixed[replacement] = fixed.pop(key)
                return fixed, replacement

        # 2) 錯誤訊息里有引號點名的欄位：payload 裡確實有就丟掉重打
        for key in _QUOTED_KEY_RE.findall(detail)[:6]:
            head = key.split(".")[-1]
            if head in payload and head not in tried and head not in _CRITICAL_KEYS:
                fixed = dict(payload)
                fixed.pop(head)
                return fixed, head

        # 3) 沒打引號的寫法（例：Unrecognized request argument: options）：掃 payload 現有欄位。
        #    順序＝自己加料的先丟，sampling 次之，response_format 最後（丟了會失去結構保證，
        #    chat_json 那一層接著改用寬鬆模式重問，不是直接放棄格式）。
        candidates = [key for key in self.caps.extra_payload if key != self.caps.token_param]
        candidates += [key for key in ("temperature", "top_p", "frequency_penalty",
                                       "presence_penalty", "response_format", "stop")
                       if key in payload]
        candidates += [key for key in payload
                       if key not in _CRITICAL_KEYS and key != self.caps.token_param
                       and key not in candidates]
        for key in candidates:
            if key in tried:
                continue
            if re.search(rf"\b{re.escape(key)}\b", detail):
                fixed = dict(payload)
                fixed.pop(key)
                return fixed, key
        return None

    # -- 送 request ---------------------------------------------------------
    def _post(self, payload: dict[str, Any], tried: tuple[str, ...] = ()) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._http().post(url, headers=headers, json=payload,
                                     timeout=self.timeout or self.caps.timeout)
        if response.status_code == 200:
            return response.json()

        detail = response.text[:800]
        healed = self._heal(payload, detail, tried)
        if healed:
            fixed, key = healed
            if self.verbose:
                print(f"    ! {self.caps.name} 不認得參數 {key}，调整后重打", flush=True)
            return self._post(fixed, tried + (key,))
        raise ProviderError(f"HTTP {response.status_code}: {detail}")

    @staticmethod
    def _content(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"回應裡沒有 choices：{str(data)[:300]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if isinstance(text, list):        # 少數端點會回 content parts
            text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
        finish = str(choices[0].get("finish_reason") or "")
        if finish == "length":
            raise ProviderError("輸出被長度截斷（finish_reason=length）；"
                                 "請調高 max_tokens 或減少一次要回的欄位）")
        return str(text or "").strip()

    def _call(self, system: str, prompt: str, images: list[Path],
              schema: type[BaseModel] | None, tag: str) -> str:
        data = self._post(self._payload(system, prompt, images, schema))
        usage = data.get("usage") or {}
        if usage:
            self.last_usage = usage
            for key, value in usage.items():
                if isinstance(value, int):
                    self.total_usage[key] = int(self.total_usage.get(key, 0)) + value
        self.calls += 1
        content = self._content(data)
        self.dump_raw(tag, content)        # 成功的也留，事後比對模型到底怎麼回
        return content

    # -- 對外：結構化輸出 ＋ 驗證 ＋ 降階重試 --------------------------------
    def chat_json(
        self,
        schema: type[T],
        system: str,
        prompt: str,
        images: list[Path] | None = None,
        tag: str = "call",
        retries: int = 2,
    ) -> T:
        """回傳已通過 pydantic 驗證的 `schema` 實體；失敗一律 ProviderError。

        第 1 次用 strict json_schema（格式由服務端保證）；之後降成寬鬆 JSON 模式，
        並把 schema 附在 prompt 尾端（不然模型只见过 schema 一次，降階等於裸打）。
        """
        last_error = "未知錯誤"
        for attempt in range(1, max(1, retries) + 1):
            strict = attempt == 1
            request_prompt = prompt if strict else prompt + format_instructions(schema)
            try:
                content = self._call(system, request_prompt, images or [],
                                     schema if strict else None, tag if strict else f"{tag}_loose")
                reply = schema.model_validate(_loads(content))
                return reply
            except json.JSONDecodeError as error:
                last_error = f"回應不是合法 JSON（{error}）"
            except ValidationError as error:
                last_error = f"回應不符合 schema（{error.error_count()} 個錯誤）"
                for item in error.errors(include_url=False)[:3]:
                    last_error += f"：{'.'.join(str(x) for x in item.get('loc') or [])}"
            except Exception as error:            # 網路／API／截斷等都要留痕
                last_error = f"{type(error).__name__}: {error}"
            if self.verbose:
                print(f"    ! {tag} 第 {attempt} 次失敗：{last_error[:200]}", flush=True)
            if attempt < retries:
                time.sleep(2)
        raise ProviderError(f"模型呼叫失敗（{tag}）：{last_error}")


def _loads(content: str) -> Any:
    """寬鬆模式偶爾會包一層 ```json 圍欄，先剝掉再 parse。"""
    text = (content or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start > 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def build_client(provider: str = "auto", model: str = "", base_url: str = "",
                 **kwargs: Any) -> LLMClient:
    """解析 env → Capability → LLMClient。`provider="auto"` 跟著 .env 走。"""
    load_env()
    resolved = auto_provider() if (provider or "auto") == "auto" else resolve_provider(provider)
    caps = CAPABILITIES[resolved]
    api_key = _azure_key() if caps.needs_key else os.getenv("OLLAMA_API_KEY", "ollama")
    if caps.needs_key and not api_key:
        raise RuntimeError(f"要用 {caps.name}，但 .env 沒有 AZURE_OPENAI_API_KEY／AZURE_API_KEY")
    base = normalize_base_url(base_url or os.getenv(
        "AZURE_OPENAI_ENDPOINT" if caps.needs_key else "OLLAMA_HOST", ""), caps)
    model_name = (model or os.getenv("AZURE_MODEL" if caps.needs_key else "OLLAMA_MODEL", "")
                  or caps.default_model)
    return LLMClient(caps=caps, model=model_name, base_url=base, api_key=api_key, **kwargs)


def _probe(argv: list[str]) -> int:
    """最小可跑的自我檢查：`python llm_provider.py --provider gpt -p "說出你的模型名"`"""
    import argparse
    from pydantic import Field, create_model

    parser = argparse.ArgumentParser(description="provider 接觸層自我檢查（不讀圖片也能跑）")
    parser.add_argument("--provider", default="auto",
                        help="auto|ollama|gpt|openai|azure（預設 auto：有 Azure 金鑰就 gpt）")
    parser.add_argument("--model", default="")
    parser.add_argument("--host", default="", help="OpenAI 相容 base_url")
    parser.add_argument("-p", "--prompt", default="用一句中文說明你是誰。")
    args = parser.parse_args(argv)

    client = build_client(args.provider, args.model, args.host)
    reply_model = create_model("ProbeReply", __base__=BaseModel,
                               answer=(str, Field(description="一句話回答")))
    started = time.time()
    reply = client.chat_json(reply_model, "你是測試用的精簡回答引擎。", args.prompt, tag="probe")
    print(f"provider={client.name} base={client.base_url}")
    print(f"answer={reply.answer}")
    print(f"用時 {time.time() - started:.1f}s，usage={client.last_usage or '（未回傅）'}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_probe(sys.argv[1:]))
