# 進度記錄：mannual_parser 重構（圖片→JSON→DB，md 只是 debug）

> 建立日期：2026-11-06　狀態：**進行中（約完成 55%）**　接手時先讀這份 + `mannual_to_md.md`

## 1. 需求（使用者原話整理）

- 舊流程 `mannual_parser.py` → `output/bbw_1/page.md` 品質不理想（OCR 錯字、整塊圖消失）。
- 人工標竿稿：`assets/bbw/bbw_1.md` + `assets/bbw/bbw_1/*.png`（已逐字校對＋彩色截圖＋bbox）。
- 規格文件：`mannual_to_md.md`（8 步流程、OCR 陷阱、圖片規則、front matter、交付驗證）。
- **要求的新架構**：先產 JSON（含文字＋關鍵圖片）→ 用**獨立的 json→md function** 渲 md
  （md 僅 debug 中繼）→ 後續只需把 JSON 轉進向量資料庫。

## 2. 已決定的架構（實作時別走樣）

```
圖片 ─[0 矯正]→ <doc_id>.page.png   ← 唯一座標系，所有 bbox 以它為準
    ─[1 OCR]─→ ocr/page_context.json（跑在同一張 page.png 上！）
    ─[2 VLM]─→ 版面與文字（閱讀順序/標題/勝負條件）
    ─[3 VLM]─→ 關鍵圖片清單（bbox/圖說/識別色/裝飾排除）
    ─[4 裁切]→ <doc_id>/<doc_id>_<img-id>.png（彩色原圖、不增強不去背）
    ─[5 產出]→ <doc_id>.json  ← 唯一真相（要進 DB 的就是它）
                ├─ json_to_md.render_markdown() → <doc_id>.md（可重渲）
                └─ page_document.to_vector_records() → <doc_id>.records.jsonl
```

- **JSON 是 source of truth**：任何「md 有、JSON 沒有」的資訊都算 bug。
- 三份新模組，職責不得互相侵入：
  | 檔 | 職責 | 不能依賴 |
  |---|---|---|
  | `page_document.py` | schema v2.0、正規化、驗證(§6)、向量記錄 | OCR/VLM/圖片 |
  | `json_to_md.py` | 純渲染 doc.json → md | OCR/VLM/PIL |
  | `mannual_parser.py` | workflow 編排（矯正/OCR/VLM/裁切/組裝） | — |
- **座標系只有一個**（舊流程最大錯位來源）：矯正後存 `page.png`，OCR 與 VLM 都吃同一張，
  所以 bbox 可直接互比、直接裁。Azure 若吃原 jpg 會落在 EXIF 直向空間（已踩過）。
- 命名／欄位調整（與標竿稿的差異，故意为之）：
  `factions` → `subjects`（加 `kind`，可容陣營/角色/欄位）、`banner_color` → `identifier_color`、
  `faction` → `subject`；md front matter 跟著新欄位名。doc_id 用檔名 `bbw_1`（非 `bbw-1`）。

## 3. 已完成

### `page_document.py`（483 行，可 import，測過）
- `SCHEMA_VERSION="2.0"`；`PAGE_TYPES / BLOCK_TYPES`（含 **image** 區塊型）/`IMAGE_RETENTION`。
- `new_document / normalize_document / normalize_block / normalize_image / clean_bbox /
  parse_image_size / save_document / load_document / add_step`。
- `validate_document(doc, base_dir)` 實作 §6：圖片 id 唯一、檔名禁中文、檔存在、caption 必備、
  **正文 image_id 集合 == images[].id 集合**（雙向）、bbox 落在 image_size 內、bbox IoU>0.2 警示、
  OCR 行數 vs 段落數 coverage、coordinate_space 未填警告。`errors_of()` 過濾錯誤。
- `to_vector_records(doc)`：切塊 = overview（標題+摘要+subjects 勝負條件/識別）→
  每個 heading 一個 section chunk（圖片以 `image_mention()` 文字融入，含識別色/視覺線索）→
  layout chunk → 每張關鍵圖一條 `chunk_type=image` 記錄。
  記錄欄位：`id / text / text_for_embedding（含【標題／小節】面包屑）/ metadata（flatten 成純量）`。
- `write_records_jsonl / read_records_jsonl / dedupe_records`（向量庫 upsert 依 id）。

### `json_to_md.py`（239 行）
- `render_front_matter`（自訂 yaml Dumper：數字陣列压成一列，bbox 可讀）、
  `render_body`（heading/paragraph/list_item/image+圖說/note/table_row，`style=bold` → `**text**`；
  未綁定的孤圖還是渲出並加註解）、`render_footer`（版面備註＋不確定處＋驗證警告 → HTML 註解）。
- `render_markdown(doc) / write_markdown(doc, path)`、CLI：
  `python json_to_md.py <doc>.json [-o out.md] [--check]`。

### `mannual_parser.py`（約 1,330 行，**流程完整、已端到端跑通**，見 §4）
基礎設施：
- 模組 docstring（流程圖＋指令）；`load_env` 從 `llm_provider` import（自己讀 .env，不拉
  python-dotenv，`build_client()` 會自動呼叫）。
- 模型接觸層已經**搬出去**：`from llm_provider import LLMClient, build_client, load_env`
  ＋ `ask_reply(client, <ReplyModel>, ...)`（見 §4.1）。本檔只剩業務邏輯，不再碰 HTTP。
- `rectify_page()`（auto/exif/none/90/180/270，存 PNG，回傳 size/applied/coordinate_space）、
  `prepare_detection_image()`（長邊 1568 JPEG）、`enhance_local()`（§3 高通）、
  `draw_grid_overlay()`（§3 標線圖，給人工定位）。
- `run_ocr(provider=paddle|azure|none)`、`_ensure_dotenv_shim()`（venv 沒裝 dotenv 的解法）、
  `polygon_to_bbox / paragraph_bbox / context_paragraphs / crop_line_image`（§1.4 放大判讀圖）。
- `denormalize_bbox`（0-1000 歸一化，也收絕對像素）、`snap_bbox_to_ink`（§3 墨韻投影微調，
  面積暴漲 4 倍退回原框）、`bbox_overlap_x`。

## 4. 狀態（2026-11-07：端到端已跑通）

### 已完成
- `mannual_parser.py`（約 1,330 行）全部 12 項待辦实裝完畢，含五個次命令
  `parse / page / md / records / validate`，旗標：`--page --doc-id --page-context --provider
  --model --host --det-long-side --ocr paddle|azure|none --ocr-unwarp --orientation --device
  --enhance/--no-enhance --review/--no-review --max-images --crop-pad --snap --grid
  --md/--no-md --records/--no-records --strict --out`。
- 端到端（`--provider gpt --page-context ocr/page_context.json`）跑 `assets/bbw/bbw_1.jpg --page 1`
  約 3.5 分鐘，`validate` **0 錯誤 0 警告**，產出 11 筆 records、5 張裁切圖。
- 與標竿 `assets/bbw/bbw_1.md` 比對：主標題「血與刃的白薔薇」、5 枚盾旗（a1 花苞／a2 盛開／
  b1 交叉雙刀／b2 長劍／b3 短劍）、灰（白薔薇）／紅（血刃）識別色、兩個勝負條件**全對**，
  且簡繁陷阱字（薔、須、擊、們、數量）這次都正確。
- 跑通後修的 4 個問題（都已在碼裡）：
  1. 兩欄頁的閱讀順序：模型會先依序吐兩欄標題再接兩欄內文 → `order_blocks_by_layout()`
     用 bbox＋`layout.column_ranges` 重排（跨欄塊按 y、各欄由左而右；與跨欄塊同高的塊一併當跨欄）。
  2. 圖片掛錯節：`bind_images_to_blocks()` 先以小節標題（`_section_bounds`）決定屬主，
     節內再用幾何決定先後，同定點依圖序（a1、a2…）排。
  3. 二次判讀把頁面那輪較好的圖說／識別色蓋掉 → 改成「只補洞」（缺才補、cues 合併去重）。
  4. 版面備註被渲染成 `- - `（模型自帶項目記號）→ compose/render 兩端都先剝記號。
- `paddleocr_pp_structure.py` 新增 `pre_rectified` 參數／`--pre-rectified`（預設由 parser 傳 True）。

### 尚待處理
1. **人工終檢**（§7）：`bbw_1.json` 的 `proofread` 仍是「尚未經人工校對」；實測內文有一處
   漏字（「不願意償還」少了「償」）這種就是機判的極限，校對後改 front matter 再 `md` 重渲。
2. 批次 `bbw_2..12`：加 shell／py 迴圈＋`--strict`，並決定多頁合併策略（整包 records 匯入、
   跨頁同名圖形的 id 衝突、`page` 排序）。
3. `vector_db.py`：把 `records.jsonl` upsert 進真正的向量庫（目前環境沒有任何向量庫可裝）。
4. 可選 `--verify-text`：逐塊裁 line 圖重轉寫（`crop_line_image()` 已就緒，還缺 schema／開關）。
5. 其他頁型實測：表格頁（`table_row`）、-component 清單頁對 `order_blocks_by_layout` 的表現。

## 4.1 模型接觸層 `llm_provider.py`（2026-11-07 重構，503 行）

**要旨：一條 OpenAI 相容路徑 ＋ pydantic 當 schema 唯一來源。** 旧的 `VisionClient`
（Ollama 走 SDK `format=<schema>`／`images=[path]`，Azure 走 REST `response_format`／
`image_url`，兩套 payload 散在 `if provider`）已整批移除。

- **一條路徑**：`build_client(provider, model, base_url)` → `LLMClient`；一律 POST
  `<base_url>/chat/completions`。Azure 吃 `AZURE_OPENAI_ENDPOINT`（已是 `/openai/v1`），
  Ollama 吃 `OLLAMA_HOST`（自動補 `/v1`）；只給根位址也會補齊。圖片一律 `image_url` data URL。
- **schema 只寫一处**：`StructureReply / VisualsReply / CropReviewReply`（pydantic，
  在 `mannual_parser.py`）。`openai_schema()` 產生 JSON Schema → 展開 `$ref/$defs` →
  白名單濾關鍵字（**`properties` 的 key 是欄位名，濾不得**）→ strict 模式補
  `additionalProperties:false` ＋ 全欄必填。回傳由同一 model 驗證（型別會自動收併、
  多餘欄位忽略、`page_type` 等枚舉錯會被擋），`ask_reply()` 再 `model_dump()` 降回 dict，
  所以下游 `normalize_*`（幾何／去重／綁小節）一行都不用改。
  **約束：Reply 欄位一律必填不給 default**（strict mode 要求），讀不到就留空字串。
- **能力表 `Capability`**：`det_long_side`（gpt 2048／ollama 1568）、`image_detail`、
  `temperature`（gpt 不送、ollama 0.1）、`token_param`（`max_completion_tokens`／
  `max_tokens`）、`extra_payload`（ollama 的 `options.num_ctx`）。新增後端＝加一列，不新增分支。
- **`_heal()` 自愈**（照錯誤訊息改 payload 重打，試過的鍵不重複試）：長度欄改名 →
  訊息點名（有引號）的欄位 → payload 現存的加料欄位（自己加的優先、sampling 次之、
  `response_format` 最後）。`model/messages` 永不當多餘參數丟。
- **重試**：第 1 次 strict `json_schema`，之後降寬鬆 JSON 模式**並把 schema 附在 prompt 尾端**
  （舊版降階時完全沒再給 schema，等於裸打）。成功的回應也留 `_debug/llm_raw/*.txt`；
  `finish_reason=length` 會被當成錯誤丟回重試而不是解析半份 JSON。`last_usage/total_usage`
  記 token（之後要算帳就有）。
- 自我檢查：`python llm_provider.py --provider gpt -p "你是誰"`（不用跑整條 pipeline；
  gpt 約 2s、ollama 約 8s，兩邊都回 structured ＋ usage）。
- 現況：`gpt`（`gpt-6-luna`）單頁 7 次呼叫約 3.5 分鐘，**本專案預設**（`.env` 有金鑰就 auto 選它）；
  `ollama` 離線可用但本頁 structure 呼叫 10 分鐘起（`qwen3-vl:8b-instruct` 跑 4K 頁）。
- ⚠ `gpt-6-luna` 不收 `temperature`，**輸出因此不可完全重現**：實測重跑一次，12 個文字塊
  相似度 0.92–1.00、`page_type` 從 `other` 變 `scoring`、旗幟 slug 從 `a1-rosebud-banner`
  變 `a1-bud`（檔名跟著變）。所以「人工終檢 §7」不是客套，且**別把 slug 當穩定 id 用**。

### 重構後的回歸（已做）
同一張頁、同一份 OCR context、同一 `--provider gpt`：block 數與 `(type, level)` 序列、
subjects、layout、other_artwork、裁切 5 張、records 11 筆、`validate` 0 錯誤 0 警告 —
**全部與重構前一致**；差異只有上面那條模型非確定性。

## 5. 環境與探測事實（已驗證，別再試錯）

- Python：`.venv/Scripts/python.exe`（3.12.3，Windows）。已裝：PIL、numpy、yaml、ollama、
  paddleocr、openai。**未裝**：python-dotenv（故有 `_ensure_dotenv_shim`）、任何向量庫
  （chromadb/qdrant/... 全缺 → 所以 DB 介面先交 `records.jsonl`）。
- Bash 指令有長度上限（單次約幾 KB），**長檔必須分段 heredoc 追加**，否則會被截斷報錯。
- Windows 的 python 看不到 git-bash 的 `/tmp`，暫存一律用專案內 `tmp_probe/`。
- Ollama 兩處可用：`http://localhost:11434`（qwen3-vl:8b-instruct、qwen3.8:27b）、
  tailscale `100.123.189.49:7401`（qwen3-vl:8b 等）。`format=<schema>` 強制 JSON 可行；
  8B 模型對本頁 5 枚盾旗的定位誤差約 1–2%（可用），但**要明確要求才肯列圖片**，
  否則會把它們寫进 text blocks（prompt 要寫死）。role 必餵 `user`，餵錯會 400 tokenize 錯。
- `assets/bbw/bbw_1.jpg`：原像素 **4000x2252**、EXIF Orientation=6。
  `exif_transpose` → 2252x4000（直），再 `rotate(90, expand=True)` → **4000x2252**，
  與人工標竿 `assets/bbw/ocr_tmp/up.png`（4000x2252）同空間 ⇒ `rectify_page(orientation="auto")`
  正好重現標竿座標系（實測 VLM bbox 鏡像對上標竿 bbox，確認無誤）。
- 標竿 `images[].bbox` == 實際裁圖尺寸（如 a1-rose bbox 330x430 == png 330x430），
  代表人工稿的 bbox 已含留白；`--crop-pad` 預設 15px 並把**裁後實際框**寫回 bbox，兩者永遠一致。
- `.env` 已收齊全部設定（範本 `.env.sample`，鍵集合由 `check_env.py` 守住）：
  `AZURE_OPENAI_ENDPOINT`＋`AZURE_OPENAI_API_KEY`＋`AZURE_MODEL`（`--provider gpt` 吃這三個）、
  `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`＋`AZURE_API_KEY`（`--ocr azure`）、
  `OLLAMA_HOST`／`OLLAMA_MODEL`／`OLLAMA_API_KEY`（`--provider ollama`，本機
  `127.0.0.1:11434` 有 `qwen3-vl:8b-instruct`、`qwen3.8:27b`）、`AZURE_ENDPOINT`（留檔，沒人讀）。
  兩套 Azure 資源同一把 key，所以 `AZURE_OPENAI_API_KEY` 直接複製 `AZURE_API_KEY`。
- **三個 `.env` 相關的坑已修**（`llm_provider.load_env`／`parse_env_line` 兩個、`01_azure_document_intelligence.py` 一個）：
  舊版不剝行尾 `#`，照 `.env.sample` 那種 `KEY=value # 註解` 一填就變成值裡含「`# 註解`」；
  舊版也永遠 `setdefault`，shell 有髒值時 `.env` 說不动它 → 新增 `ENV_FORCE=1`／`load_env(force=True)`。
  預設仍是 shell 贏（`export` 覆寫要用），空值 `KEY=` 一律不覆蓋。
  另：`01_azure_document_intelligence.py` 原本硬 `from dotenv import load_dotenv`，
  但本專案沒裝 python-dotenv → 單獨跑必 `ModuleNotFoundError`（經 mannual_parser 跑才有 shim），
  已改成 ImportError 時退回 `llm_provider.load_env`。
- **時間**：PaddleOCR PP-StructureV3 在 4000x2252 上、開著去彎曲＝單頁 15-20 分鐘（絕大部分
  是 UVDoc／方向判斷）。朝向本來就由 [0] 做，所以 `run_ocr` 預設傳 `pre_rectified=True`；
  同一頁要重跑判讀就 `--page-context output/<doc>/ocr/page_context.json` 整個跳過 OCR。
  GPT 端「結構＋圖片＋5 張裁切復判」＝7 次呼叫約 3.5 分鐘。
- 舊產物定位（別搞混，§8）：`output/bbw_1/`＝Paddle 草稿（錯字多、paragraphs 沒 bbox）、
  `output_azure/page_context.json`＝Azure 草稿（bbox 是 **EXIF 直向空間**，吃原 jpg 才會錯位）、
  `boardgame_output/`＝舊 pipeline 失敗產物。Paddle 對本頁 `layout_blocks` 是空的 →
  文字 bbox 只能靠 Azure 或 VLM。

## 6. 接手後先跑這三條

```bash
PY=.venv/Scripts/python.exe
$PY -c "import page_document, json_to_md, mannual_parser; print('modules ok')"
$PY mannual_parser.py --help                       # 五個子命令都在
$PY mannual_parser.py parse assets/bbw/bbw_1.jpg --page 1     --page-context output/bbw_1/ocr/page_context.json   # 沿用 OCR，約 3.5 分鐘
$PY mannual_parser.py validate output/bbw_1/bbw_1.json  # 現況：通過／0 個問題
```

已跑出的成品在 `output/bbw_1/`（`bbw_1.json`／`bbw_1.md`／`bbw_1/*.png`／`bbw_1.records.jsonl`），
舊 Paddle 草稿搬到 `output/_legacy/`。

## 7. 尚待確認的設計問題（使用者還沒點頭）

- `records.jsonl` 之後要進哪個庫？（Chroma / Azure AI Search / pgvector…）
  決定 `vector_db.py` 適配器要不要現在寫。
- 是否要 `--verify-text`（逐塊放大重轉寫；`crop_line_image()` 已就緒，缺 schema／開關與預算）。
- 多頁批次（bbw_1..12）與「同一遊戲多頁合并成一份文件（doc_id + page）」的彙入策略。
