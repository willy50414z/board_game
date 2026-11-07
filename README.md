# 規則書頁面 → 文件 JSON（→ 向量資料庫）

**正式入口只有一個：`mannual_parser.py`。** 流程是「圖片 → JSON」，JSON 才是唯一真相；
md 只是 debug 預覽（可隨時用 JSON 重渲），向量資料庫吃的是 `records.jsonl`。

```
圖片 ─[0 朝向矯正]→ <doc>.page.png      ← 唯一座標系，bbox 一律以此為準
     ─[1 OCR]──────→ ocr/page_context.json（草稿文字＋bbox）
     ─[2 VLM]──────→ 版面與文字（閱讀順序、標題、勝負條件）
     ─[3 VLM]──────→ 關鍵圖片（bbox、圖說、識別色、裝飾排除）
     ─[4 裁切]─────→ <doc_id>/*.png（彩色原解析度，不增強不去背）＋逐張二次判讀
     ─[5 產出]─────→ <doc>.json ─┬─ json_to_md  → <doc>.md（debug）
                                 └─ to_vector_records → <doc>.records.jsonl（匯 DB）
     ─[6 驗證]─────→ §6 交付前檢查（id 唯一、檔名、bbox 在圖內、block↔image 對齊…）
```

## 快速開始

```bash
# 全流程（.env 有 Azure 金鑰就自動用 GPT；要本機模型加 --provider ollama）
python mannual_parser.py parse assets/bbw/bbw_1.jpg --page 1

# 其他常用组合
python mannual_parser.py parse assets/bbw/bbw_2.jpg --page 2 --provider ollama --no-review
python mannual_parser.py page  assets/bbw/bbw_1.jpg            # 只矯正＋OCR，先確認座標系
python mannual_parser.py md    output/bbw_1/bbw_1.json --check # 人工修 JSON 後重渲 md
python mannual_parser.py records output/bbw_1/bbw_1.json       # 只重產 DB 記錄
python mannual_parser.py validate output/bbw_1/bbw_1.json --strict
```

常用旗標：`--ocr paddle|azure|none`、`--page-context <json>`（沿用既有 OCR，省 20 分鐘）、
`--provider auto|ollama|gpt`、`--model`、`--no-review`（跳過逐張判讀）、`--grid`（產 100px 標線圖）、
`--strict`（驗證有錯就非零 exit，批次用）。

## `.env`

```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1   # GPT（OpenAI 相容端點）
AZURE_API_KEY=...                                                     # 或 AZURE_OPENAI_API_KEY
AZURE_MODEL=gpt-6-luna
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...                              # 走 --ocr azure 時要
OLLAMA_HOST=http://localhost:11434                                    # 走 --provider ollama 時要
OLLAMA_MODEL=qwen3-vl:8b-instruct
```

## 模型 provider 層（`llm_provider.py`）

**只有一條 request 路徑**：所有後端都打 OpenAI 相容的 `<base_url>/chat/completions`
（Azure 的 `/openai/v1`、Ollama 的 `/v1`），所以業務碼裡沒有 `if provider`。

- **schema 唯一來源＝pydantic**：`mannual_parser.py` 的 `StructureReply / VisualsReply /
  CropReviewReply` 產生送出去的 JSON Schema（自動展開 `$ref`、濾成各家收得的關鍵字），
  收回來也由同一個 model 驗證，型別不符／少欄位會直接重問而不是靜默塞預設值。
  欄位說明（`Field(description=...)`）只寫一处。
- **差異登记在 `Capability`**（`gpt`／`ollama` 各一列）：偵測圖長邊、要不要 `detail`、
  要不要送 `temperature`、長度欄叫 `max_tokens` 還是 `max_completion_tokens`、額外 payload。
- **被唸參數會就地自愈**：`_heal()` 照錯誤訊息改名／丟參數後重打（`max_tokens`↔
  `max_completion_tokens`、不收 `temperature`、不認得新增欄位、schema 被拒時改走寬鬆模式）。
- **重試**：第 1 次 `json_schema(strict)`，之後降成寬鬆 JSON 模式並把 schema 附在 prompt 尾端；
  每次呼叫（含失敗、含降階）都留 `_debug/llm_raw/*.txt`。
- 自我檢查不用跑整條 pipeline：`python llm_provider.py --provider gpt -p "你是誰"`。

| provider | 預設模型 | 筆記 |
|---|---|---|
| `gpt` | `AZURE_MODEL` | `gpt-6-luna` 只收預設 `temperature`（＝輸出會飄，校對必做）、長度欄是 `max_completion_tokens`、`json_schema` 只吃 JSON Schema 子集（`minItems` 等會被濾掉） |
| `ollama` | `OLLAMA_MODEL` | 離線可用；`/v1` 沒有 `num_ctx`，改由外層 `options` 帶（被拒就丟掉重打）。本頁 structure 呼叫動輒 10 分鐘以上，批次以 `gpt` 為主 |

## 產物

| 檔案 | 誰在用 |
|---|---|
| `output/<doc>/<doc>.json` | **唯一真相**，進 DB 前的資料；人工校對就改這裡 |
| `output/<doc>/<doc>.records.jsonl` | 向量資料庫 upsert（`id` 穩定：`<doc>#overview`、`#S01`、`#layout`、`#img-<id>`） |
| `output/<doc>/<doc>.md` | debug 預覽，`mannual_parser.py md` 可重渲，不進 DB |
| `output/<doc>/<doc>.page.png` | 矯正後的唯一座標系基準圖 |
| `output/<doc>/<doc>/*.png` | 關鍵圖片裁切（檔名 `<doc>_<小節><序>_<english>.png`） |
| `output/<doc>/ocr/page_context.json` | OCR 草稿（同一座標系） |
| `output/<doc>/_debug/` | structure/visuals/llm_raw/grid 等除錯與追查用 |

## 三個一定要知道的坑

1. **OCR 只是草稿。** 簡繁／近似字（薔/蔷、須/须、擊/撃、們/们）與裝飾字體都會錯，
   文字一律以圖片判讀為準；`proofread` 欄沒改成人工校對前，別當作可交付版本。
2. **PaddleOCR 預設不再去彎曲**（`--ocr-unwarp` 才開）。朝向由 [0] 處理，4K 照片開去彎曲
   單頁會花 15-20 分鐘；同一頁重跑請用 `--page-context` 沿用 OCR。
3. **自動產物仍需人工終檢**（`mannual_to_md.md` §7）：圖說與相鄰元素的對比、切圖有沒有切到
   邻居、顏色識別色是否一致，機判只是「先給你一個 80 分的草稿」。

---

## Legacy MVP scripts（01／03）

以下三個腳本是最早的單頁 MVP 樣本，`mannual_parser.py` 只借用前兩個產 `page_context.json`
（contract 相同），其余邏輯請以 `mannual_parser.py` 為準。

These three scripts form a small, testable MVP for **one still photo of one rulebook page**.

- `01_azure_document_intelligence.py`: Azure Document Intelligence `prebuilt-layout` -> `page_context.json`
- `02_paddleocr_pp_structure.py`: local PaddleOCR PP-StructureV3 -> same `page_context.json` contract
- `03_ollama_qwen3vl_analyze.py`: reads either output -> Ollama Qwen3-VL structured analysis JSON

## 0. Prerequisites

- Python 3.10+ recommended.
- Ollama is running and the local model tag exists:

```bash
ollama pull qwen3-vl:8b-instruct
ollama list
```

If your local tag differs, set `OLLAMA_MODEL` accordingly.

## 1. Azure path

Use a separate virtual environment if desired:

```bash
python -m venv .venv-azure
source .venv-azure/bin/activate
python -m pip install --upgrade pip
python -m pip install azure-ai-documentintelligence ollama
```

Set credentials:

```bash
export DOCUMENTINTELLIGENCE_ENDPOINT='https://<resource>.cognitiveservices.azure.com/'
export DOCUMENTINTELLIGENCE_API_KEY='<key>'
export OLLAMA_HOST='http://localhost:11434'
export OLLAMA_MODEL='qwen3-vl:8b-instruct'
```

Run:

```bash
python 01_azure_document_intelligence.py ./samples/page_001.jpg --out ./work/azure
python 03_ollama_qwen3vl_analyze.py ./work/azure/page_context.json --out ./work/azure/analysis.json
```

The Azure sample requests `prebuilt-layout` and `FIGURES` output. It writes paragraph IDs, layout text, extracted figure crops, and the raw DI result. The actual image is still sent to Qwen3-VL so the model can interpret the complete page context.

## 2. Local PaddleOCR path

PP-StructureV3 needs PaddleOCR's `doc-parser` extras and an appropriate PaddlePaddle inference engine. Create a separate environment; follow the official PaddlePaddle installation matrix for your operating system and CPU/GPU backend. For the first pass on an AMD/ROCm host, use CPU to avoid assuming Paddle's GPU backend works with ROCm.

```bash
python -m venv .venv-paddle
source .venv-paddle/bin/activate
python -m pip install --upgrade pip
# Install a PaddlePaddle CPU wheel using the official PaddlePaddle install instructions.
python -m pip install "paddleocr[doc-parser]" pillow ollama
```

Run:

```bash
python paddleocr_pp_structure.py ./samples/page_001.jpg --out ./work/paddle --device cpu
python 03_ollama_qwen3vl_analyze.py ./work/paddle/page_context.json --out ./work/paddle/analysis.json
```

The sample attempts to crop layout blocks labelled as images/figures/charts. If none are detected, Qwen receives the full page and assesses it at page level.

## 3. What the scripts do / do not do

- They operate on one captured still image per run. They do not record video, detect page changes, or run live OCR.
- Azure and PaddleOCR both emit a common `page_context.json`, so you can compare both extraction methods using the same Qwen script.
- Qwen produces a draft only. It does not directly publish or delete records.
- Paragraph links are checked against IDs returned by the OCR/layout stage; invalid IDs are removed and marked for review.
- OCR/layout figure detection varies by image quality and model. A missing crop does not mean the printed page has no illustration.

## Official references

- Azure Document Intelligence Python SDK: https://learn.microsoft.com/python/api/overview/azure/ai-documentintelligence-readme
- Azure Layout model: https://learn.microsoft.com/azure/ai-services/document-intelligence/prebuilt/layout
- PaddleOCR PP-StructureV3: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html
- PaddleOCR installation: https://www.paddleocr.ai/latest/en/version3.x/installation.html
- Ollama Qwen3-VL: https://ollama.com/library/qwen3-vl
# board_game
