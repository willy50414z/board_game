# Board-game rulebook MVP samples

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
