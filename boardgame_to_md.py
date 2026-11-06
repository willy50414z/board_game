#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桌遊說明書圖片 -> Markdown PoC
流程：OpenCV 前處理 -> PaddleOCR PP-StructureV3 -> Ollama Vision -> Markdown

安裝（請依官方文件先安裝與目前 Python/CUDA/ROCm 相容的 PaddlePaddle）：
    pip install -U opencv-python requests "paddleocr[doc-parser]"

執行：
    python boardgame_to_md.py --input 20261031_112151.jpg

常用參數：
    --ollama-url http://localhost:7401
    --model qwen2.5vl:7b
    --rotation ccw
    --split-ratio 0.50
    --scale 2
    --skip-vision
    --no-enhance

注意：
- 這是單頁/單張照片 PoC，切頁比例需視照片調整。
- OCR Markdown 是文字辨識結果；Vision 描述是輔助理解，不能視為已驗證的遊戲規則。
- 程式會保留原始彩色頁面及 OCR 輸出，方便人工核對。
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import requests

import os

# 修正 PaddleX / oneDNN PIR 相容性問題
# 必須在 import paddleocr / paddlex 之前設定
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
SYSTEM_PROMPT = """你是桌遊說明書的視覺版面分析助手。
請只描述圖片中能直接觀察到的資訊，不要推測看不清楚的規則。
不要重抄圖片中的完整段落文字；OCR 會另外處理文字。
請著重：
1. 版面區域及其相對位置
2. 卡牌、道具、圖示、玩家版圖等視覺元素
3. 圖片與附近標題/標籤可能存在的對應關係
4. 表格或流程圖的視覺結構（不要猜測模糊數字）
5. 無法確認的內容請明確標示「無法辨識」或 uncertain
請以繁體中文回答，並輸出符合 JSON 格式的物件。
JSON 欄位：
{
  "page_summary": "本頁可觀察到的整體摘要",
  "regions": [
    {
      "region": "區域名稱或位置",
      "visual_elements": ["可見元素"],
      "labels_seen": ["清楚可讀的短標籤；看不清楚就留空"],
      "relationships": ["有把握的圖文/位置關係"],
      "uncertain": ["無法確認的項目"]
    }
  ],
  "tables_or_diagrams": ["表格或流程圖的可見結構描述"],
  "verification_notes": ["建議人工核對的地方"]
}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="使用 PaddleOCR + Ollama Vision 將桌遊說明書圖片轉成 Markdown"
    )
    p.add_argument("--input", required=True, help="輸入圖片路徑")
    p.add_argument("--output", default="boardgame_output", help="輸出資料夾")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--model", default="qwen2.5vl:7b", help="Ollama Vision 模型名稱")
    p.add_argument("--rotation", choices=["none", "cw", "ccw", "180"], default="ccw",
                   help="圖片旋轉方向；此範例照片預設逆時針 90 度")
    p.add_argument("--split-ratio", type=float, default=0.50,
                   help="左右頁分割比例，0~1；預設 0.50")
    p.add_argument("--scale", type=float, default=2.0, help="頁面放大倍率")
    p.add_argument("--skip-vision", action="store_true", help="略過 Ollama Vision")
    p.add_argument("--no-enhance", action="store_true", help="略過 CLAHE 對比增強")
    p.add_argument("--timeout", type=int, default=300, help="Ollama HTTP timeout 秒數")
    p.add_argument("--language", default="ch", help="PaddleOCR 語言，繁體中文通常使用 ch")
    return p.parse_args()


def rotate_image(img, direction: str):
    if direction == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if direction == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if direction == "180":
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def enhance_for_ocr(img):
    """保留彩色原圖；只對 OCR 版本做溫和局部對比增強。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def save_preprocessed_pages(
    input_path: Path, assets_dir: Path, rotation: str,
    split_ratio: float, scale: float, enhance: bool
) -> list[dict[str, Path]]:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"無法讀取圖片：{input_path}")

    image = rotate_image(image, rotation)
    h, w = image.shape[:2]
    if not 0.05 <= split_ratio <= 0.95:
        raise ValueError("--split-ratio 必須介於 0.05 到 0.95")

    split_x = int(w * split_ratio)
    raw_pages = [
        ("page_01", image[:, :split_x]),
        ("page_02", image[:, split_x:])
    ]

    pages: list[dict[str, Path]] = []
    for name, raw in raw_pages:
        if scale != 1:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            raw = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=interp)

        original_path = assets_dir / f"{name}_color.jpg"
        ocr_path = assets_dir / f"{name}_ocr.jpg"

        cv2.imwrite(str(original_path), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ocr_img = enhance_for_ocr(raw) if enhance else raw
        cv2.imwrite(str(ocr_path), ocr_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        pages.append({"name": Path(name), "color": original_path, "ocr": ocr_path})

    return pages


def build_ocr_pipeline(language: str):
    # 環境變數必須在載入 PaddleX 前設定
    import os
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

    from paddleocr import PPStructureV3

    return PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_table_recognition=True,
        enable_mkldnn=False,
    )


def run_ocr(pipeline, image_path: Path, output_dir: Path, stem: str) -> tuple[str, list[str]]:
    """執行 PP-StructureV3，盡可能保存 JSON/Markdown，回傳可用的 Markdown。"""
    result = pipeline.predict(str(image_path))
    saved_files: list[str] = []
    md_parts: list[str] = []

    for idx, item in enumerate(result):
        # 官方結果物件通常支援 save_to_json / save_to_markdown。
        if hasattr(item, "save_to_json"):
            try:
                item.save_to_json(save_path=str(output_dir / f"{stem}_result_{idx}.json"))
                saved_files.append(f"{stem}_result_{idx}.json")
            except Exception as exc:
                print(f"[WARN] 儲存 OCR JSON 失敗：{exc}")

        if hasattr(item, "save_to_markdown"):
            try:
                item.save_to_markdown(save_path=str(output_dir))
            except Exception as exc:
                print(f"[WARN] PP-StructureV3 Markdown 匯出失敗：{exc}")

        # 嘗試從結果物件取出 Markdown 字串或 dict 內容。
        data: Any = None
        if hasattr(item, "json"):
            try:
                data = item.json
                if callable(data):
                    data = data()
            except Exception:
                data = None
        if data is None and isinstance(item, dict):
            data = item

        if isinstance(data, str):
            md_parts.append(data)
        elif isinstance(data, dict):
            text = extract_text_from_structure(data)
            if text:
                md_parts.append(text)
            else:
                md_parts.append("```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```")

    # 某些版本會自行在 output_dir 產生 markdown 檔，讀取對應檔案。
    candidates = sorted(output_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        newest = candidates[0]
        try:
            exported = newest.read_text(encoding="utf-8")
            if exported.strip():
                md_parts.insert(0, exported)
        except Exception:
            pass

    markdown = "\n\n".join(x.strip() for x in md_parts if x and x.strip())
    if not markdown:
        markdown = (
            "> PP-StructureV3 已執行，但目前版本未能透過通用 API 取得 Markdown 字串。\n"
            "> 請查看同資料夾的 OCR JSON/PP-StructureV3 匯出檔。"
        )
    return markdown, saved_files


def extract_text_from_structure(obj: Any) -> str:
    """從常見 PP-Structure JSON 欄位遞迴抽取文字，作為 Markdown 匯出失敗時的 fallback。"""
    found: list[str] = []

    def walk(x: Any):
        if isinstance(x, dict):
            for key, value in x.items():
                if key.lower() in {"text", "rec_text", "content"} and isinstance(value, str):
                    value = value.strip()
                    if value and value not in found:
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)

    walk(obj)
    return "\n\n".join(found)


def ollama_vision(
    image_path: Path, model: str, base_url: str, timeout: int
) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "請分析這張桌遊說明書頁面的視覺版面與圖文關係。",
                "images": [image_b64],
            },
        ],
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 模型若回傳 JSON 外框文字，嘗試擷取第一個 JSON object。
        match = re.search(r"\{.*\}", content, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_response": content, "parse_error": "Vision 回應不是有效 JSON"}


def vision_to_markdown(data: dict[str, Any]) -> str:
    lines = ["### 圖片版面分析（Ollama Vision；需人工核對）", ""]
    summary = data.get("page_summary")
    if summary:
        lines += [str(summary), ""]

    for region in data.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        lines += [f"#### {region.get('region', '未命名區域')}"]
        for label, key in [
            ("可見元素", "visual_elements"),
            ("可辨識標籤", "labels_seen"),
            ("圖文/位置關係", "relationships"),
            ("不確定項目", "uncertain"),
        ]:
            values = region.get(key) or []
            if values:
                lines.append(f"**{label}**")
                lines.extend(f"- {v}" for v in values)
        lines.append("")

    for title, key in [
        ("表格或圖示結構", "tables_or_diagrams"),
        ("人工核對事項", "verification_notes"),
    ]:
        values = data.get(key) or []
        if values:
            lines += [f"#### {title}"]
            lines.extend(f"- {v}" for v in values)
            lines.append("")

    if data.get("raw_response"):
        lines += ["```text", str(data["raw_response"]), "```", ""]
    return "\n".join(lines).strip()


def write_final_markdown(
    output_dir: Path, pages: list[dict[str, Path]],
    ocr_results: dict[str, str], vision_results: dict[str, dict[str, Any]]
):
    lines = [
        "# 桌遊說明書（自動轉換初稿）",
        "",
        "> 本文件由 PaddleOCR PP-StructureV3 與 Ollama Vision 自動產生，尚未經人工校對。",
        "> 遊戲規則、數字、否定詞、玩家人數、卡牌數量及勝利條件請務必對照原圖確認。",
        "",
        "## 原始頁面",
        "",
    ]

    for page in pages:
        name = page["name"].name
        color_rel = f"assets/{page['color'].name}"
        lines += [f"### {name}", "", f"![{name}]({color_rel})", ""]

    lines += ["---", "", "## OCR 文字與表格辨識結果", ""]

    for page in pages:
        name = page["name"].name
        lines += [f"### {name}", "", ocr_results.get(name, "> 無 OCR 結果"), ""]

    lines += ["---", "", "## 圖片與版面理解", ""]

    for page in pages:
        name = page["name"].name
        lines += [f"### {name}", ""]
        if name in vision_results:
            lines += [vision_to_markdown(vision_results[name]), ""]
        else:
            lines += ["> 已略過 Vision 分析。", ""]

    lines += [
        "---",
        "",
        "## 校對清單",
        "",
        "- [ ] 逐段核對 OCR 文字與原始圖片",
        "- [ ] 核對所有數字、玩家人數及卡牌數量",
        "- [ ] 核對「不、不得、不能、必須」等關鍵詞",
        "- [ ] 核對表格列欄及每個儲存格",
        "- [ ] 核對角色名稱、圖片與陣營的對應",
        "- [ ] 核對圖片引用路徑是否有效",
        "- [ ] 將 Vision 推測或不確定內容改為人工確認的描述",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    assets_dir = output_dir / "assets"
    ocr_dir = output_dir / "ocr"
    vision_dir = output_dir / "vision"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入圖片：{input_path}")
    if not 0.1 <= args.scale <= 5:
        raise ValueError("--scale 建議介於 0.1 到 5")

    for directory in (output_dir, assets_dir, ocr_dir, vision_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/4] 圖片前處理與左右頁切割...")
    pages = save_preprocessed_pages(
        input_path, assets_dir, args.rotation,
        args.split_ratio, args.scale, not args.no_enhance
    )
    print(f"      已產生 {len(pages)} 頁")

    print("[2/4] 載入 PaddleOCR PP-StructureV3...")
    try:
        pipeline = build_ocr_pipeline(args.language)
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR PP-StructureV3 初始化失敗。請確認已安裝相容的 "
            "PaddlePaddle 與 paddleocr[doc-parser]。\n"
            f"原始錯誤：{exc}"
        ) from exc

    ocr_results: dict[str, str] = {}
    for i, page in enumerate(pages, start=1):
        name = page["name"].name
        print(f"[3/4] OCR {name} ({i}/{len(pages)})...")
        try:
            md, _ = run_ocr(pipeline, page["ocr"], ocr_dir, name)
            ocr_results[name] = md
            (ocr_dir / f"{name}.md").write_text(md, encoding="utf-8")
        except Exception as exc:
            print(f"[ERROR] {name} OCR 失敗：{exc}")
            traceback.print_exc()
            ocr_results[name] = f"> OCR 失敗：{exc}"

    vision_results: dict[str, dict[str, Any]] = {}
    if not args.skip_vision:
        print("[4/4] Ollama Vision 分析頁面...")
        for i, page in enumerate(pages, start=1):
            name = page["name"].name
            print(f"      Vision {name} ({i}/{len(pages)})...")
            try:
                data = ollama_vision(
                    page["color"], args.model, args.ollama_url, args.timeout
                )
                vision_results[name] = data
                (vision_dir / f"{name}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"[ERROR] {name} Vision 失敗：{exc}")
                vision_results[name] = {
                    "verification_notes": [f"Vision 呼叫失敗：{exc}"]
                }
    else:
        print("[4/4] 已依參數略過 Ollama Vision。")

    write_final_markdown(output_dir, pages, ocr_results, vision_results)

    manifest = {
        "input": str(input_path),
        "created_at": datetime.now().astimezone().isoformat(),
        "rotation": args.rotation,
        "split_ratio": args.split_ratio,
        "scale": args.scale,
        "ollama_model": None if args.skip_vision else args.model,
        "pages": [p["name"].name for p in pages],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n完成！")
    print(f"Markdown：{output_dir / 'README.md'}")
    print(f"輸出資料夾：{output_dir}")
    print("請先檢查左右頁切割是否正確，再校對 README.md。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n執行失敗：{error}", file=sys.stderr)
        sys.exit(1)
