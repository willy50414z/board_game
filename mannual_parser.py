#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""說明書圖片 -> 頁面文件 JSON ->（向量資料庫記錄 / md 預覽）。

對應 mannual_to_md.md 的自動化版本。分工：

    圖片 ──[0 矯正]──> <doc>.page.png（單一座標系，bbox 一律以此為準）
         ──[1 OCR]───> ocr/page_context.json（文字草稿＋區塊 bbox）
         ──[2 VLM]───> 版面與文字（閱讀順序、標題、勝負條件）
         ──[3 VLM]───> 關鍵圖片清單（bbox、圖說、顏色識別、裝飾排除）
         ──[4 裁切]──> <doc_id>/*.png（彩色原圖原解析度，不增強不去背）
         ──[5 產出]──> <doc>.json          ← 唯一真相，要進 DB 的就是它
                        ├─ json_to_md      → <doc>.md（debug 預覽，可重渲）
                        └─ to_vector_records → <doc>.records.jsonl（DB 匯入）

重點：**座標系只有一個**。OCR 與 VLM 都跑在同一張矯正後的 page.png 上，
所以產物裡的 bbox 可直接互相比對、裁切（舊流程最大的錯位來源）。

用法：
    python mannual_parser.py parse assets/bbw/bbw_1.jpg --page 1
    python mannual_parser.py parse assets/bbw/bbw_2.jpg --page 2 --ocr azure --no-review
    python mannual_parser.py md       output/bbw_1/bbw_1.json      # 人工修 JSON 後重渲 md
    python mannual_parser.py records  output/bbw_1/bbw_1.json      # 只重產 DB 記錄
    python mannual_parser.py validate output/bbw_1/bbw_1.json      # §6 驗證
    python mannual_parser.py page     assets/bbw/bbw_1.jpg         # 只做矯正＋OCR
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, ConfigDict, Field
from llm_provider import T as ReplyT
from llm_provider import LLMClient, build_client, load_env
from page_document import (
    add_step, clean_bbox, load_document, new_document, normalize_document,
    save_document, to_vector_records, validate_document, write_records_jsonl,
)

DET_LONG_SIDE = 1568          # 偵測圖長邊 fallback（各家預設值見 llm_provider.Capability）
EPS = 1e-6


# ---------------------------------------------------------------------------
# 0. 矯正：建立唯一座標系（§1.1）
# ---------------------------------------------------------------------------

def rectify_page(image_path: Path, out_path: Path, orientation: str = "auto") -> dict[str, Any]:
    """EXIF 矯正＋必要旋轉，存成 Lossless PNG，之後所有 bbox 都以它為準。

    orientation:
      auto  EXIF 矯正後若是直圖，轉成橫向（本專案翻拍照多为橫向版面）
      exif  只做 EXIF 矯正
      none  完全不處理（直接原像素）
      90/180/270  EXIF 矯正後再逆時針轉 N 度
    """
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw) if orientation != "none" else raw
        applied: list[str] = [] if orientation == "none" else ["exif_transpose"]

        if orientation == "auto" and image.height > image.width:
            image = image.rotate(90, expand=True)
            applied.append("rotate_90_ccw(auto→landscape)")
        elif orientation in ("90", "180", "270"):
            angle = int(orientation)
            image = image.rotate(angle, expand=True)
            applied.append(f"rotate_{angle}_ccw")

        width, height = image.size
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(out_path, format="PNG", compress_level=3)

    return {
        "path": out_path,
        "width": width,
        "height": height,
        "orientation": orientation,
        "applied": applied or ["none"],
        "coordinate_space": f"{width}x{height} 矯正後{'橫向' if width >= height else '直向'}圖（bbox 即此座標系）",
    }


def prepare_detection_image(page_path: Path, out_path: Path, long_side: int = DET_LONG_SIDE) -> Path:
    """VLM 偵測用縮圖（JPEG）。bbox 用 0-1000 歸一化，縮圖不會降低定位精度。"""
    with Image.open(page_path) as page:
        image = page.convert("RGB")
        if max(image.size) > long_side:
            scale = long_side / float(max(image.size))
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=92)
    return out_path


def enhance_local(image_path: Path, out_path: Path, k: float = 3.0, offset: int = 110,
                  max_side: int = 0) -> Path:
    """局部正規化高通濾波（§3）：深底白字／灰部陰影的細節會浮出來。"""
    import numpy as np

    with Image.open(image_path) as handle:
        gray = handle.convert("L")
    if max_side and max(gray.size) > max_side:
        scale = max_side / float(max(gray.size))
        gray = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
                           Image.LANCZOS)
    blurred = gray.filter(ImageFilter.GaussianBlur(30))
    delta = np.asarray(gray, dtype="int32") - np.asarray(blurred, dtype="int32")
    enhanced = np.clip(delta * k + offset, 0, 255).astype("uint8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced, mode="L").save(out_path)
    return out_path


def draw_grid_overlay(page_path: Path, out_path: Path, step: int = 100) -> Path:
    """疊 100px 座標標線圖（§3）：人工定位／校正 bbox 時最好用。"""
    from PIL import ImageDraw

    with Image.open(page_path) as handle:
        canvas = handle.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font_size = max(16, canvas.width // 120)
    for x in range(0, canvas.width, step):
        draw.line([(x, 0), (x, canvas.height)], fill=(255, 0, 0), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, canvas.height, step):
        draw.line([(0, y), (canvas.width, y)], fill=(255, 0, 0), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 1. OCR：只做草稿與幾何參考（§1.3、§2）
# ---------------------------------------------------------------------------

def _ensure_dotenv_shim() -> None:
    """01_azure_document_intelligence.py 依賴 python-dotenv；沒有就用本機 .env 載入器代理。"""
    try:
        import dotenv  # noqa: F401
        return
    except ModuleNotFoundError:
        import types

        shim = types.ModuleType("dotenv")
        shim.load_dotenv = lambda *a, **k: load_env()
        sys.modules["dotenv"] = shim


def polygon_to_bbox(polygon: Any) -> list[int] | None:
    """Azure DI 的 polygon（[x1,y1,x2,y2,...]）→ 外接框。"""
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 4:
        return None
    xs = [float(v) for i, v in enumerate(polygon) if i % 2 == 0]
    ys = [float(v) for i, v in enumerate(polygon) if i % 2 == 1]
    if not xs or not ys:
        return None
    return clean_bbox([min(xs), min(ys), max(xs), max(ys)])


def paragraph_bbox(paragraph: dict[str, Any]) -> list[int] | None:
    """段落 bbox 來源優先序：bbox > boundingRegions.polygon > regions[].polygon。"""
    bbox = clean_bbox(paragraph.get("bbox"))
    if bbox:
        return bbox
    for key in ("bounding_regions", "regions"):
        for region in paragraph.get(key) or []:
            bbox = polygon_to_bbox(region.get("polygon"))
            if bbox:
                return bbox
    return None


def run_ocr(page_path: Path, ocr_dir: Path, provider: str, device: str = "cpu",
            unwarp: bool = False) -> dict[str, Any]:
    """對**矯正後的 page.png** 跑 OCR，回傳 page_context（與 page.png 同座標系）。

    朝向已在 [0] 處理過，所以預設叫 Paddle 跳過自己的方向判斷與去彎曲
    （4000px 等級的照片，這兩步佔掉大半時間）；版面真的會彎才加 --ocr-unwarp。
    """
    if provider in ("none", "skip"):
        return {"source_image": str(page_path), "ocr_provider": "none", "paragraphs": [], "ocr_text": ""}

    ocr_dir.mkdir(parents=True, exist_ok=True)
    # PP-StructureV3 的 oneDNN/PIR 相容性開關需在 import 前設定
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

    if provider == "paddle":
        module = importlib.import_module("paddleocr_pp_structure")
        context_path = module.parse_pic_to_json(page_path, ocr_dir, device, pre_rectified=not unwarp)
    elif provider == "azure":
        _ensure_dotenv_shim()
        load_env()
        module = importlib.import_module("01_azure_document_intelligence")
        context_path = module.run(page_path, ocr_dir)
    else:
        raise ValueError(f"不支援的 OCR provider: {provider}")

    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    context["source_image"] = str(page_path.resolve())   # 強制對準矯正圖
    return context


def context_paragraphs(context: dict[str, Any]) -> list[dict[str, Any]]:
    """抽出給 VLM 的 OCR 段落精簡版（id/text/bbox），避免把整個 context 塞進 prompt。"""
    payload: list[dict[str, Any]] = []
    for paragraph in context.get("paragraphs") or []:
        text = str(paragraph.get("text") or "").strip()
        if not text:
            continue
        payload.append({
            "id": str(paragraph.get("id") or f"P{len(payload) + 1:04d}"),
            "text": text[:600],
            "bbox": paragraph_bbox(paragraph),
        })
    return payload


def crop_line_image(page_path: Path, bbox: list[int], out_path: Path, scale: float = 2.5) -> Path:
    """逐行／逐塊放大判讀用圖（§1.4）：放大後再做局部正規化，不要一次看整頁。"""
    with Image.open(page_path) as handle:
        page = handle.convert("RGB")
    pad = max(4, int(0.006 * max(page.size)))
    x1, y1, x2, y2 = bbox
    box = (max(0, x1 - pad), max(0, y1 - pad), min(page.width, x2 + pad), min(page.height, y2 + pad))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise ValueError(f"bbox 太小，無法放大判讀: {bbox}")
    crop = page.crop(box)
    if scale != 1:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gray = crop.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(30))
    import numpy as np
    delta = np.asarray(gray, dtype="int32") - np.asarray(blurred, dtype="int32")
    enhanced = np.clip(delta * 3 + 112, 0, 255).astype("uint8")
    Image.fromarray(enhanced, mode="L").save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 座標：VLM 回傳 0-1000 歸一化 → 矯正圖像素；再用墨韻投影微調（§3）
# ---------------------------------------------------------------------------

def denormalize_bbox(bbox: Any, width: int, height: int) -> list[int] | None:
    """模型若偷傳絕對像素也自動辨識（最大值 <= 1000 才視為歸一化）。"""
    values = clean_bbox(bbox)
    if not values:
        return None
    if max(values) <= 1000:
        x1, y1, x2, y2 = values
        values = [
            int(round(x1 / 1000 * width)), int(round(y1 / 1000 * height)),
            int(round(x2 / 1000 * width)), int(round(y2 / 1000 * height)),
        ]
    values = clean_bbox(values)
    if not values:
        return None
    x1, y1, x2, y2 = values
    box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    return box if box[2] > box[0] and box[3] > box[1] else None


def snap_bbox_to_ink(page_path: Path, bbox: list[int], grow: float = 0.06) -> list[int]:
    """把 bbox 邊界吸附到實際「有墨」的範圍。

    用局部正規化後的高通訊號做 row/column 投影，比 OCR 框穩定（§3）；
    只允許在原框外擴 grow 比例內調整，避免整頁被吸進來。
    """
    import numpy as np

    with Image.open(page_path) as handle:
        page = handle.convert("L")
    width, height = page.size
    pad = int(max(page.size) * grow)
    x1, y1, x2, y2 = bbox
    win = (max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad))
    if win[2] - win[0] < 16 or win[3] - win[1] < 16:
        return bbox

    window = page.crop(win)
    blurred = window.filter(ImageFilter.GaussianBlur(max(3, min(window.size) // 12)))
    delta = np.abs(np.asarray(window, dtype="int32") - np.asarray(blurred, dtype="int32"))
    threshold = max(16.0, float(np.percentile(delta, 95)) * 0.45)
    mask = delta > threshold
    if int(mask.sum()) < 40:
        return bbox

    columns = np.flatnonzero(mask.sum(axis=0) > 1)
    rows = np.flatnonzero(mask.sum(axis=1) > 1)
    if columns.size == 0 or rows.size == 0:
        return bbox

    snapped = [
        win[0] + int(columns.min()), win[1] + int(rows.min()),
        win[0] + int(columns.max()) + 1, win[1] + int(rows.max()) + 1,
    ]
    snapped = clean_bbox(snapped) or bbox
    # 吸附後面積暴漲代表吸到相鄰元素，退回原框
    def area(box: list[int]) -> int:
        return max(1, (box[2] - box[0]) * (box[3] - box[1]))

    if area(snapped) > area(list(bbox)) * 4:
        return bbox
    return snapped


def bbox_overlap_x(a: list[int], b: list[int]) -> float:
    """兩框在 X 軸重疊比例（判斷圖形屬於哪一欄／哪個小節）。"""
    left, right = max(a[0], b[0]), min(a[2], b[2])
    if right <= left:
        return 0.0
    widths = min(a[2] - a[0], b[2] - b[0])
    return (right - left) / float(max(widths, 1))


# ---------------------------------------------------------------------------
# 2/3. 模型回覆的形狀：pydantic model 是 schema 的【唯一來源】
#
# 送出去用的 JSON Schema 由 `model.model_json_schema()` 產生（llm_provider 會濾成
# 各家收得的關鍵字），收回來也由同一個 model 驗證——欄位說明只寫在这邊一处。
# 約束：回覆模型的欄位一律【必填、不給 default】，才符合 strict mode；
#      「圖上讀不到」用空字串表達，不要用 None/Optional。
# ---------------------------------------------------------------------------

class _Reply(BaseModel):
    """回覆模型共同設定：多出來的欄位忽略（寬鬆模式難保模型不加料）。"""

    model_config = ConfigDict(extra="ignore")


BBox = list[float]        # 0-1000 歸一化 [x_min, y_min, x_max, y_max]
_BBOX_NOTE = "0-1000 歸一化 [x_min, y_min, x_max, y_max]（相對你看到的圖片寬高 ×1000）"

PageType = Literal["cover", "components", "setup", "turn_rules", "scoring",
                   "example", "reference", "credits", "other", "unclear"]
BlockType = Literal["heading", "paragraph", "list_item", "table_row"]
Importance = Literal["critical", "useful", "decorative"]
Retention = Literal["keep", "review", "not_needed"]


class SubjectReply(_Reply):
    key: str = Field(description="頁面上的代號，如 a、b")
    name: str
    kind: str = Field(description="陣營／角色／欄位／物品…")
    win_condition: str = Field(description="獲勝條件原文；圖上讀不到就留空字串")
    identifier: str = Field(description="如何辨識（含顏色），如：灰色縞帶盾旗上的白薔薇")


class BlockReply(_Reply):
    type: BlockType
    level: int = Field(description="heading 層級 1=主標題 2=區塊 3=小節；非 heading 填 0")
    text: str = Field(description="對著圖片逐字確認過的正確文字（不是 OCR 原文）")
    source_paragraph_ids: list[str] = Field(description="對應的 OCR 段落 id")
    bbox: BBox = Field(description=_BBOX_NOTE)


class ColumnRangeReply(_Reply):
    label: str
    x_min: float
    x_max: float
    description: str


class LayoutReply(_Reply):
    columns: int
    column_ranges: list[ColumnRangeReply]
    reading_order: str = Field(description="例：由上而下，先左欄 a 再右欄 b")
    block_counts: str = Field(description="每欄有幾個文字塊／幾個圖形，例：a 欄 2 圖、b 欄 3 圖")


class StructureReply(_Reply):
    page_type: PageType
    title: str = Field(description="頁面主標題（以圖片為準，不是 OCR）")
    summary: str = Field(description="本頁在講什麼，2-3 句，只寫圖上讀得到的")
    subjects: list[SubjectReply]
    blocks: list[BlockReply]
    layout: LayoutReply
    uncertain_points: list[str] = Field(description="看不清楚或可能判讀錯的地方")


class VisualReply(_Reply):
    slug_en: str = Field(description="小寫英文短名，例 rose、crossed-blades（檔名用，禁中文）")
    caption: str = Field(description="短圖說，例：盛開的白薔薇（白薔薇卡片）")
    alt: str = Field(description="完整 alt：讓沒看到圖的人也讀得懂它在規則裡的角色")
    section: str = Field(description="所屬小節標題（與頁面文字一致）")
    subject: str = Field(description="它代表什麼（陣營名／角色／物品名）")
    identifier_color: str = Field(description="識別用底色／縞帶色，例：紅（血刃識別色）")
    non_text_cues: list[str] = Field(description="顏色、箭頭方向、序號、數量等，與同款元素的差異")
    importance: Importance
    retention: Retention
    bbox: BBox = Field(description=_BBOX_NOTE)


class VisualsReply(_Reply):
    visual_elements: list[VisualReply]
    other_artwork: list[str] = Field(
        description="純裝飾（花邊、標題裝飾、分隔線、頁碼），註明是裝飾且未截圖")
    layout_notes: str = Field(description="版面備註：欄位 x 範圍、每欄元素數、分隔線位置等")


class CropReviewReply(_Reply):
    object: str = Field(description="它是什麼，一句話")
    caption: str = Field(description="給資料庫的短圖說（<=24 字）")
    identifier_color: str = Field(description="用於識別的底色／縞帶／邊框顏色；看不出來留空")
    non_text_cues: list[str] = Field(description="形狀、箭頭指向、序號、數量、與同款圖形的差異")
    is_standalone: bool = Field(description="是否為完整獨立元素（false＝切到相鄰元素或只是局部）")
    is_decorative: bool = Field(description="是否純裝飾（花邊／分隔線／背景紋理）")


def ask_reply(client: LLMClient, reply_model: type[ReplyT], system: str, prompt: str,
              images: list[Path], tag: str) -> dict[str, Any]:
    """呼叫＋驗證，再降回 dict：下游 normalize_*（幾何、去重、綁小節）吃的仍是 dict。"""
    reply = client.chat_json(reply_model, system, prompt, images=images, tag=tag)
    return reply.model_dump()



# ---------------------------------------------------------------------------
# Prompt：把 mannual_to_md.md 的規則餵給模型（§2、§3、§4、§7）
# ---------------------------------------------------------------------------

OCR_TRAPS = (
    "本專案實際踩過的 OCR 陷阱（不可照抄草稿）：\n"
    "- 簡繁／近似字：們/们、營/营、須/须、擊/撃、薔/蔷、刃/刀、數量/数量\n"
    "- 裝飾字體易被誤判（標題「血與刃的白薔薇」曾被讀成「血照刃的白普薇」）\n"
    "- 深底白字會被閾值判成背景；低分數亂字常是圖示局部被當成文字\n"
    "- 標點原圖怎麼寫就怎麼存，不要自行統一（、 vs ，、半形 (a) vs 全形 （a））"
)

STRUCTURE_SYSTEM_PROMPT = f"""你是桌遊規則書的頁面解析引擎。輸入是一張【已矯正朝向】的規則書頁面照片，外加 OCR 草稿。

最高原則：讀你的輸出必須等於讀實體頁。規則常藏在圖示、顏色、箭頭、序號裡，只有文字＝資訊遺漏。

{OCR_TRAPS}

【任務一：閱讀順序】
1. 先判斷版面（單欄／雙欄／網格）。中文由上而下；多欄先讀完左欄再讀右欄。
2. 輸入的 OCR 段落陣列順序【不可信】，只用它取文字內容與 id；順序以圖片為準。
3. 同一語意塊被 OCR 切成多段時合併成一塊，並引用所有相關 id。

【任務二：blocks（依閱讀順序輸出頁面全部文字）】
- heading：level 1＝頁面主標題、2＝區塊標題（如「勝利目標」）、3＝小節標題（如「a) 白薔薇陣營」）
- paragraph／list_item／table_row：正文
- text 欄要是你【對著圖片逐字確認過】的正確文字，不是 OCR 原文；OCR 錯了就直接修正
- 每塊都給 bbox（該文字區塊在頁面上的位置）

【任務三：subjects】該頁的結構化主體（陣營／角色／欄位／物品）。
name、win_condition、identifier 必須能在圖上讀到；圖上沒有就留空字串，不要臆造規則。
identifier 請寫出「怎麼辨識」，包括顏色（例：灰色縞帶盾旗上的白薔薇）。

【任務四】layout 描述真實版面；不確定之處寫進 uncertain_points，不要猜。
嚴格依照 JSON Schema 輸出，不要有任何額外解釋。
""".strip()


VISUALS_SYSTEM_PROMPT = f"""你是桌遊規則書的「關鍵圖片」標註引擎。你標出來的每一個 bbox，之後會被照著從【彩色原圖】裁切入庫。

最高原則：讀你的輸出必須等於讀實體頁。規則常藏在圖示與配色裡。

【必須裁進資料庫（importance=critical/useful, retention=keep）】
所有承載規則資訊的獨立圖形：圖示、徽章、盾旗、卡牌、角色、兵器、配件照、
流程圖、配置圖、示意圖、表格內的圖。

【不要裁（importance=decorative, retention=not_needed）】
純裝飾：花邊、標題裝飾、分隔線、頁碼、背景紋理 → 改寫進 other_artwork，並寫明「純裝飾、未截圖」。

【關鍵：顏色經常是識別線索】
實例：同一款盾形旗靠縞帶底色區分陣營（灰＝白薔薇、紅＝血刃）。
- 靠顏色區分的元素，把顏色寫進 identifier_color 與 non_text_cues，並在 alt 講清楚。
- 裁切一定要用彩色原圖（系統會自動做），所以你看到的顏色就是判讀依據。

【有方向／箭頭／序號的圖】在 non_text_cues 寫出指向與順序（例：箭頭由花苞指向白薔薇），
否則日後只剩圖沒有順序，規則就斷線。

【判斷「是元素還是分隔線」】有同款外框、且高度明顯超出鄰近線條的，是元素，不要漏掉。
（實例：長劍盾旗比相鄰盾旗長一倍，差點被當成欄間分隔線。）

{OCR_TRAPS}

【bbox 規則】
- 0-1000 歸一化 [x_min, y_min, x_max, y_max]，含整個構圖（含頂部與底部尖端）。
- 寧可多留一點背景，也不要切到相鄰元素；同一排的同款圖形要一個框一個元素，不可合併。
- slug_en 用簡潔小寫英文（rose、bud、longsword、dagger、crossed-blades），檔名會用它。
- section 填該圖所屬的小節標題（例：「a) 白薔薇陣營」），subject 填它代表什麼。
嚴格依照 JSON Schema 輸出，不要有任何額外解釋。
""".strip()

CROP_REVIEW_SYSTEM_PROMPT = """你正在逐張核對「從規則書裁出來的獨立圖形」。輸入只有這一張裁切圖（彩色原圖裁切）。

只描述你【看得見】的：
1. object：它是什麼（一句話，繁體中文）
2. caption：給資料庫的短圖說（<=24 字）
3. identifier_color：用於識別的底色／縞帶／邊框主要顏色，以及它可能代表誰（例：紅（血刃識別色））
4. non_text_cues：形狀、箭頭指向、序號、數量、與同款圖形的差異
5. is_standalone：是否完整獨立（切到相鄰元素或只是半個元素 → false）
6. is_decorative：是否純裝飾（花邊、分隔線、背景紋理 → true）

不要猜規則數字；看不清就說不清。嚴格依照 JSON Schema 輸出。
""".strip()


# ---------------------------------------------------------------------------
# 2. 版面與文字分析
# ---------------------------------------------------------------------------

def analyze_structure(
    client: LLMClient,
    page_info: dict[str, Any],
    det_path: Path,
    paragraphs: list[dict[str, Any]],
    provider: str,
    enhanced_path: Path | None = None,
) -> dict[str, Any]:
    images = [det_path] + ([enhanced_path] if enhanced_path else [])
    hint = (
        "\n（共 2 張圖：第 1 張彩色原圖、第 2 張同版面高通增強版；深色底白字請多參考第 2 張，"
        "但兩者座標一致。）" if enhanced_path else ""
    )
    prompt = "\n".join([
        "請分析這張桌遊說明書頁面，依系統指示回傳結構化資料。",
        f"OCR 來源：{provider}（矯正後尺寸 {page_info['width']}x{page_info['height']}）",
        f"OCR 段落共 {len(paragraphs)} 則；順序不可信，bbox 可能是 null：",
        json.dumps(paragraphs, ensure_ascii=False),
        "" if paragraphs else "（OCR 沒有可用段落，請完全以圖片判讀。）",
        "提醒：blocks[].text 請填你對照圖片確認過的正確文字；OCR 有錯字就直接修正。" + hint,
    ])
    return ask_reply(client, StructureReply, STRUCTURE_SYSTEM_PROMPT, prompt, images, "structure")


def analyze_visuals(
    client: LLMClient,
    page_info: dict[str, Any],
    det_path: Path,
    structure: dict[str, Any],
    enhanced_path: Path | None = None,
) -> dict[str, Any]:
    sections = []
    for block in structure.get("blocks") or []:
        if str(block.get("type")) == "heading" and str(block.get("text") or "").strip():
            sections.append(str(block["text"]).strip())
    images = [det_path] + ([enhanced_path] if enhanced_path else [])
    prompt = "\n".join([
        "請列出這頁所有需要入庫的關鍵圖片（含圖示、徽章、卡牌、兵器、流程圖），並給出 bbox。",
        f"頁面標題：{structure.get('title', '')}",
        f"已判讀的小節（section 欄請沿用這些名稱，勿自創）：{json.dumps(sections, ensure_ascii=False)}",
        f"版面：{json.dumps(structure.get('layout') or {}, ensure_ascii=False)}",
        "特別注意：同款圖形若靠顏色區分陣營／身分，務必寫進 identifier_color 與 non_text_cues。",
        "同一排的同款圖形要一個框一個元素，分別列出（例：長劍、匕首、交叉雙刀各一框）。",
    ])
    return ask_reply(client, VisualsReply, VISUALS_SYSTEM_PROMPT, prompt, images, "visuals")


def review_crop(client: LLMClient, crop_path: Path) -> dict[str, Any]:
    return ask_reply(
        client, CropReviewReply, CROP_REVIEW_SYSTEM_PROMPT,
        "這是從規則書頁面裁出來的一個獨立圖形（彩色原圖裁切，未增強）。請照 schema 描述它。",
        [crop_path], f"review_{crop_path.stem[:40]}",
    )


# ---------------------------------------------------------------------------
# 4. 裁切關鍵圖片（彩色原圖、原解析度、不增強不去背）＋ 綁進正文
# ---------------------------------------------------------------------------

def safe_slug(value: str, fallback: str = "img") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug[:40] or fallback


def section_code(section: str, index: int) -> str:
    """小節標題 → 欄位代號：「a) 白薔薇陣營」→ a；找不到用 i1、i2（§4.4 檔名規則）。"""
    match = re.match(r"\s*([a-zA-Z])\s*[).、]", str(section or "").strip())
    if match:
        return match.group(1).lower()
    digits = re.match(r"\s*(\d+)", str(section or "").strip())
    if digits:
        return digits.group(1)
    return f"i{index + 1}"


def _column_index(bbox: list[int], width: int, column_ranges: list[dict[str, Any]]) -> int:
    """依中心 x 決定屬於第幾欄（先讀左欄再讀右欄，圖片順序才會跟標竿稿一致）。"""
    if not column_ranges:
        return 0
    center = ((bbox[0] + bbox[2]) / 2.0) / max(1, width) * 1000.0
    for index, column in enumerate(column_ranges):
        try:
            if float(column.get("x_min", 0)) <= center <= float(column.get("x_max", 1000)):
                return index
        except (TypeError, ValueError):
            continue
    return 0


def crop_visuals(
    page_path: Path,
    page_info: dict[str, Any],
    visuals: list[dict[str, Any]],
    doc_id: str,
    out_dir: Path,
    *,
    crop_pad: int = 15,
    snap: bool = True,
    column_ranges: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """回傳 (images, other_artwork, warnings)。images 已依「欄位 → 上下」排序並帶檔名與 bbox。"""
    width, height = page_info["width"], page_info["height"]
    images_dir = out_dir / doc_id
    warnings_: list[str] = []
    other_artwork: list[str] = []
    prepared: list[dict[str, Any]] = []

    for visual in visuals:
        caption = str(visual.get("caption") or "").strip()
        importance = str(visual.get("importance") or "").strip()
        retention = str(visual.get("retention") or "review")

        if importance == "decorative" or retention == "not_needed":
            other_artwork.append(f"{caption or '裝飾元素'}（{visual.get('section') or '本頁'}，純裝飾，未截圖）")
            continue

        bbox = denormalize_bbox(visual.get("bbox"), width, height)
        if not bbox:
            warnings_.append(f"圖片「{caption}」bbox 無效或太小，已略過：{visual.get('bbox')}")
            continue
        if snap:
            bbox = snap_bbox_to_ink(page_path, bbox)
        bbox = clean_bbox([
            max(0, bbox[0] - crop_pad), max(0, bbox[1] - crop_pad),
            min(width, bbox[2] + crop_pad), min(height, bbox[3] + crop_pad),
        ]) or bbox
        if (bbox[2] - bbox[0]) < 24 or (bbox[3] - bbox[1]) < 24:
            warnings_.append(f"圖片「{caption}」裁切區過小（{bbox}），可能是模型幻覺，已略過")
            continue

        prepared.append({**visual, "bbox": bbox, "_column": _column_index(bbox, width, column_ranges or [])})

    prepared.sort(key=lambda item: (item["_column"], item["bbox"][1], item["bbox"][0]))

    images: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    used_ids: set[str] = set()
    for item in prepared:
        code = section_code(str(item.get("section") or ""), len(images))
        counters[code] = counters.get(code, 0) + 1
        order = counters[code]
        slug = safe_slug(str(item.get("slug_en") or ""))
        image_id = f"{code}{order}-{slug}"
        while image_id in used_ids:
            image_id = f"{image_id}-{len(used_ids) + 1}"
        used_ids.add(image_id)

        file_name = f"{doc_id}_{image_id}.png"
        images_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(page_path) as handle:
            page = handle.convert("RGB")
            page.crop(tuple(item["bbox"])).save(images_dir / file_name, format="PNG")

        images.append({
            "id": image_id,
            "file": f"{doc_id}/{file_name}",
            "caption": str(item.get("caption") or "").strip(),
            "alt": str(item.get("alt") or item.get("caption") or "").strip(),
            "section": str(item.get("section") or "").strip(),
            "subject": str(item.get("subject") or "").strip(),
            "identifier_color": str(item.get("identifier_color") or "").strip(),
            "non_text_cues": list(item.get("non_text_cues") or []),
            "importance": str(item.get("importance") or "").strip(),
            "retention": "keep" if str(item.get("retention")) in ("keep", "review") else "review",
            "bbox": item["bbox"],
            "_label": f"{code}-{order}",
        })

    return images, other_artwork, warnings_


def review_crops(
    client: LLMClient,
    images: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_images: int = 8,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """逐張裁切圖二次判讀（§1.4、§4.1）：補識別色／視覺線索，並踢掉其實是裝飾的。"""
    kept: list[dict[str, Any]] = []
    artwork: list[str] = []
    warnings_: list[str] = []

    for index, image in enumerate(images):
        crop_path = out_dir / str(image.get("file") or "")
        if index >= max_images:
            warnings_.append(f"圖片超過 {max_images} 張，其餘未做二次判讀（--max-images 可調）")
            kept.append(image)
            continue
        if not crop_path.is_file():
            warnings_.append(f"找不到裁切檔，跳過二次判讀：{image.get('file')}")
            kept.append(image)
            continue

        try:
            review = review_crop(client, crop_path)
        except Exception as error:
            warnings_.append(f"{image.get('id')} 二次判讀失敗：{error}")
            kept.append(image)
            continue

        if review.get("is_decorative"):
            artwork.append(f"{image.get('caption') or image.get('id')}（{image.get('section') or '本頁'}，"
                           f"二次判讀為純裝飾，已刪除截圖）")
            crop_path.unlink(missing_ok=True)
            warnings_.append(f"{image.get('id')} 二次判讀為純裝飾，已移除")
            continue

        # 頁面圖那輪看得到全部元素（有得比），圖說與識別色通常比較準；
        # 裁切圖這輪的定位是「驗證＋補細節」，所以只補洞，不整包蓋掉。
        if not str(image.get("caption") or "").strip():
            image["caption"] = str(review.get("caption") or "").strip()
        if not image.get("alt"):
            image["alt"] = image["caption"]
        if not str(image.get("identifier_color") or "").strip():
            image["identifier_color"] = str(review.get("identifier_color") or "").strip()
        cues = [str(c).strip() for c in review.get("non_text_cues") or [] if str(c).strip()]
        merged = list(image.get("non_text_cues") or [])
        merged += [c for c in cues if c not in merged]
        image["non_text_cues"] = merged[:8]
        image["verified"] = True
        if review.get("is_standalone") is False:
            warnings_.append(f"{image.get('id')} 疑似非獨立元素（切到相鄰物或只是局部），bbox 需人工確認")
            image["retention"] = "review"
        kept.append(image)

    return kept, artwork, warnings_


def _heading_key(text: str) -> str:
    """標題比對用的歸零形式：去空白、統一括號、丟標點（「a) 白薔薇陣營」↔「a）白薔薇陣營」）。"""
    value = re.sub(r"[\s　]+", "", str(text or ""))
    value = value.replace("）", ")").replace("（", "(").replace("．", ".")
    return re.sub(r"[()\[\].、,，:：;；]", "", value)


def _section_bounds(blocks: list[dict[str, Any]], section: str) -> tuple[int, int] | None:
    """小節標題對上的話回傳 (起, 未)；圖只能插在這一節裡面，不然會掛到隔壁節。"""
    key = _heading_key(section)
    if not key:
        return None
    for index, block in enumerate(blocks):
        if block.get("type") != "heading":
            continue
        heading_key = _heading_key(block.get("text"))
        if not heading_key or (key not in heading_key and heading_key not in key):
            continue
        level = int(block.get("level") or 1)
        for follower in range(index + 1, len(blocks)):
            nxt = blocks[follower]
            if nxt.get("type") == "heading" and int(nxt.get("level") or 1) <= level:
                return index, follower
        return index, len(blocks)
    return None


def _column_zone(bbox: list[int] | None, width: int,
                 column_ranges: list[dict[str, Any]]) -> int | str:
    """这一块屬於哪一欄；跨欄（或只有一欄）回傳 'cross'。column_ranges 是 0-1000 歸一化。"""
    if not bbox or width <= 0 or len(column_ranges) < 2:
        return "cross"
    zones: list[tuple[int, int]] = []
    for column in column_ranges:
        try:
            left = int(float(column.get("x_min")) / 1000 * width)
            right = int(float(column.get("x_max")) / 1000 * width)
        except (TypeError, ValueError):
            continue
        if right > left:
            zones.append((left, right))
    if len(zones) < 2:
        return "cross"
    left, right = bbox[0], bbox[2]
    if right - left > 0.55 * width:
        return "cross"
    covered = [i for i, (a, b) in enumerate(zones) if min(right, b) - max(left, a) > 0]
    if len(covered) != 1:
        return "cross"
    return covered[0]


def _vertical_overlap_ratio(a: list[int], b: list[int]) -> float:
    """兩個 bbox 在垂直方向的重疊度（判斷「同一行」用）。"""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(a[3] - a[1], b[3] - b[1]))


def order_blocks_by_layout(blocks: list[dict[str, Any]], width: int,
                           column_ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """幾何重排閱讀順序：多欄時模型常把兩欄的標題先依序輸出、再接兩欄內文。

    順序 ＝ 跨欄區（按 y）→ 各欄内容由左而右（各欄按 y）→ 剩下的跨欄區。
    與跨欄塊同高的塊（例如橫跨兩欄的「勝利目標」標題＋說明那行）一起當跨欄處理，
    否則會被捲進其中一欄，讀起來就跳欄了。
    """
    if len(blocks) < 3 or len(column_ranges) < 2:
        return blocks

    def top_of(block: dict[str, Any]) -> int:
        bbox = block.get("bbox")
        return int(bbox[1]) if bbox else 0

    def left_of(block: dict[str, Any]) -> int:
        bbox = block.get("bbox")
        return int(bbox[0]) if bbox else 0

    columns: dict[int, list[dict[str, Any]]] = {}
    cross: list[dict[str, Any]] = []
    for block in blocks:
        zone = _column_zone(block.get("bbox"), width, column_ranges)
        (cross if zone == "cross" else columns.setdefault(zone, [])).append(block)
    if len(columns) < 2:
        return blocks

    promoted = [block for column in list(columns.values()) for block in column
                if block.get("bbox") and any(_vertical_overlap_ratio(block["bbox"], c["bbox"]) >= 0.5
                                             for c in cross if c.get("bbox"))]
    for block in promoted:
        for column in columns.values():
            if block in column:
                column.remove(block)
                break
        cross.append(block)
    columns = {zone: column for zone, column in columns.items() if column}
    if len(columns) < 2:
        return blocks

    band_top = min(top_of(b) for column in columns.values() for b in column)
    head = sorted((b for b in cross if top_of(b) < band_top), key=lambda b: (top_of(b), left_of(b)))
    tail = sorted((b for b in cross if top_of(b) >= band_top), key=lambda b: (top_of(b), left_of(b)))
    ordered = head + [b for zone in sorted(columns)
                      for b in sorted(columns[zone], key=lambda b: (top_of(b), left_of(b)))] + tail
    return ordered if len(ordered) == len(blocks) else blocks


def bind_images_to_blocks(blocks: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[str]:
    """把圖片插到「同欄、位於其上方最近」的文字區塊之後；找不到就掛到最後。"""
    warnings_: list[str] = []
    insertions: list[tuple[int, dict[str, Any]]] = []

    for image in images:
        bbox = image.get("bbox")
        bounds = _section_bounds(blocks, str(image.get("section") or ""))   # 小節決定屬主
        anchor = bounds[0] if bounds else None                              # 預設：標題正下方
        start, stop = bounds or (0, len(blocks))
        if bbox:
            candidates = []
            for index in range(start, stop):
                block_bbox = blocks[index].get("bbox")
                if not block_bbox:
                    continue
                if index == start and blocks[index].get("type") == "heading":
                    # 小節標題永遠能當錨點（圖通常就緊貼在標題下方）
                    candidates.append((0, index))
                    continue
                if bbox_overlap_x(bbox, block_bbox) < 0.3:
                    continue
                if block_bbox[3] > bbox[3]:       # 文字在圖下方，不能當錨點
                    continue
                candidates.append((block_bbox[3], index))
            if candidates:
                anchor = max(candidates)[1]      # 節內的先後順序交給幾何決定

        image_block = {
            "type": "image",
            "text": "",
            "image_id": image.get("id"),
            "caption": "圖 {}｜{}".format(
                image.get("_label") or image.get("id"),
                image.get("caption") or image.get("alt") or ""),
            "bbox": bbox,
        }
        if anchor is None:
            insertions.append((len(blocks), image_block))
            warnings_.append(f"圖片 {image.get('id')} 找不到所屬文字小節，先掛在頁尾（請人工於 JSON 調整 blocks 順序）")
        else:
            insertions.append((anchor + 1, image_block))

    # 同一個定點要插多張圖時依圖序（a1、a2…）排，重跑結果才會一致
    insertions.sort(key=lambda pair: (pair[0], str(pair[1].get("image_id") or "")), reverse=True)
    for position, image_block in insertions:
        blocks.insert(position, image_block)
    return warnings_


# ---------------------------------------------------------------------------
# 5. 組裝文件 JSON（唯一真相）
# ---------------------------------------------------------------------------

def relative_path(path: Path | str, start: Path | None = None) -> str:
    """能轉專案相對路徑就轉，避免 DB 裡塞滿本機絕對路徑。"""
    path = Path(path)
    try:
        relative = os.path.relpath(path, start or Path.cwd())
    except ValueError:
        return str(path).replace(os.sep, "/")
    return str(path).replace(os.sep, "/") if relative.startswith("..") else relative.replace(os.sep, "/")


def _clean_text(text: str) -> str:
    """模型偶爾會把 OCR 的 ## 一起抄進來；層級由 block['level'] 決定。"""
    return re.sub(r"^[\s#>*-]+", "", str(text or "")).strip()


def compose_layout_notes(structure: dict[str, Any], visuals_layout_notes: str) -> str:
    layout = structure.get("layout") or {}
    lines: list[str] = []
    columns = layout.get("columns")
    if columns:
        lines.append(f"- 版面欄數：{columns}；閱讀順序：{layout.get('reading_order') or '由上而下、先左後右'}")
    for column in layout.get("column_ranges") or []:
        lines.append(
            f"- 欄位 {column.get('label', '')}：x≈{column.get('x_min')}-{column.get('x_max')}"
            f"（0-1000 歸一化）{('：' + str(column.get('description'))) if column.get('description') else ''}"
        )
    if layout.get("block_counts"):
        lines.append(f"- 每欄元素數：{layout['block_counts']}")
    if str(visuals_layout_notes or "").strip():
        if lines:
            lines.append("- 圖片版面：")
        for line in str(visuals_layout_notes).strip().splitlines():
            line = re.sub(r"^\s*[-*·•]\s*", "", line).strip()
            if line:
                lines.append(f"- {line}")
    lines.append("- 本檔由 mannual_parser.py 自動產出，未經人工逐字校對；文字以圖片判讀為準。")
    return "\n".join(lines)


def build_document(
    *,
    doc_id: str,
    page_number: int | None,
    source_image: Path,
    page_info: dict[str, Any],
    ocr_context: dict[str, Any],
    structure: dict[str, Any],
    visuals: dict[str, Any],
    images: list[dict[str, Any]],
    other_artwork: list[str],
    warnings_: list[str],
    model: str,
) -> dict[str, Any]:
    width, height = page_info["width"], page_info["height"]
    title = _clean_text(structure.get("title") or "")

    blocks: list[dict[str, Any]] = []
    for raw in structure.get("blocks") or []:
        text = _clean_text(raw.get("text"))
        block_type = str(raw.get("type") or "paragraph")
        if not text:
            continue
        level = raw.get("level") or 0
        if block_type == "heading":
            try:
                level = int(level)
            except (TypeError, ValueError):
                level = 2
            if level <= 1 and (text == title or (title and title in text)):
                continue          # 主標題放 front matter，正文不重複
            level = level or 2
        blocks.append({
            "type": block_type,
            "level": level,
            "text": text,
            "bbox": denormalize_bbox(raw.get("bbox"), width, height),
            "source_paragraph_ids": [str(x) for x in raw.get("source_paragraph_ids") or []],
        })

    blocks = order_blocks_by_layout(blocks, width,
                                    (structure.get("layout") or {}).get("column_ranges") or [])
    warnings_ += bind_images_to_blocks(blocks, images)

    doc = new_document(
        doc_id,
        title=title,
        page=page_number,
        source_image=relative_path(source_image),
        language="zh-TW",
    )
    doc["page_type"] = str(structure.get("page_type") or "unclear")
    doc["type"] = f"規則書第 {page_number} 頁" if page_number else ""
    doc["summary"] = _clean_text(structure.get("summary"))
    doc["page_image"] = f"{doc_id}.page.png"
    doc["image_size"] = f"{width}x{height}"
    doc["capture"] = (
        f"手機翻拍（原件 {Path(source_image).name} "
        f"{page_info.get('raw_size', '')}，朝向處理：{'、'.join(page_info.get('applied') or ['none'])}）"
    )
    doc["ocr_provider"] = str(ocr_context.get("ocr_provider") or "")
    doc["subjects"] = [
        {
            "key": str(s.get("key") or "").strip(),
            "name": _clean_text(s.get("name")),
            "kind": str(s.get("kind") or "").strip(),
            "win_condition": _clean_text(s.get("win_condition")),
            "identifier": _clean_text(s.get("identifier")),
        }
        for s in structure.get("subjects") or [] if isinstance(s, dict) and _clean_text(s.get("name"))
    ]
    doc["images"] = images
    doc["other_artwork"] = other_artwork
    doc["blocks"] = blocks
    doc["layout_notes"] = compose_layout_notes(structure, visuals.get("layout_notes", ""))
    doc["image_source"] = {
        "file": relative_path(source_image),
        "coordinate_space": f"{page_info['coordinate_space']}；來源 {doc['page_image']}",
        "processing": "EXIF 矯正後存 Lossless PNG；裁圖為原解析度原色，未增強、未去背",
    }
    doc["quality"] = {
        "needs_review": True,
        "uncertain_points": [str(x) for x in structure.get("uncertain_points") or []],
        "warnings": list(dict.fromkeys(warnings_)),
    }
    doc["ocr_draft"] = {
        "provider": doc["ocr_provider"],
        "paragraph_count": len(ocr_context.get("paragraphs") or []),
        "text": str(ocr_context.get("ocr_text") or "")[:8000],
    }
    doc["provenance"]["generator"] = "mannual_parser.py"
    doc["provenance"]["model"] = model
    return normalize_document(doc)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

@dataclass
class ParseOptions:
    page: int | None = None
    doc_id: str = ""
    provider: str = "auto"                    # auto | gpt（Azure Foundry）| ollama | openai
    ocr: str = "paddle"                       # paddle | azure | none
    page_context: Path | None = None          # 直接沿用既有 OCR 結果
    orientation: str = "auto"
    model: str = ""
    host: str = ""
    enhance: bool = True                      # 多送一張高通增強圖（深底白字很有感）
    review: bool = True                       # 逐張裁切圖二次判讀
    max_images: int = 8
    crop_pad: int = 15
    snap: bool = True
    grid: bool = False
    write_md: bool = True
    write_records: bool = True
    strict: bool = False
    device: str = "cpu"
    det_long_side: int = 0                    # 0 = 看 provider 給預設值
    ocr_unwarp: bool = False             # 頁面真的會彎才開（很慢）


def log(message: str) -> None:
    print(message, flush=True)


def safe_doc_id(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")
    return value or "page"


def parse_page(image_path: Path, out_dir: Path, options: ParseOptions) -> dict[str, Any]:
    load_env()
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    doc_id = safe_doc_id(options.doc_id or image_path.stem)

    client = build_client(options.provider, options.model, options.host,
                          raw_dir=debug_dir / "llm_raw")
    model = client.name
    det_long_side = options.det_long_side or client.det_long_side

    # [0] 矯正：建立唯一座標系
    with Image.open(image_path) as probe:
        raw_size = f"{probe.width}x{probe.height}"
    page_path = out_dir / f"{doc_id}.page.png"
    page_info = rectify_page(image_path, page_path, options.orientation)
    page_info["raw_size"] = raw_size
    log(f"[0] 矯正後 {page_info['width']}x{page_info['height']}"
        f"（{'、'.join(page_info['applied'])}）→ {page_path.name}")

    det_path = prepare_detection_image(page_path, out_dir / "page_det.jpg", det_long_side)
    enhanced_path = (enhance_local(page_path, debug_dir / "page_enh.jpg", max_side=det_long_side)
                     if options.enhance else None)
    if options.grid:
        log(f"[0] 標線圖 → {draw_grid_overlay(page_path, debug_dir / 'grid.png')}")

    # [1] OCR 草稿
    if options.page_context:
        ocr_context = json.loads(Path(options.page_context).read_text(encoding="utf-8"))
        ocr_context["source_image"] = str(page_path.resolve())
        log(f"[1] 沿用 OCR：{options.page_context}")
    else:
        ocr_context = run_ocr(page_path, out_dir / "ocr", options.ocr, options.device,
                            options.ocr_unwarp)
    paragraphs = context_paragraphs(ocr_context)
    log(f"[1] OCR {ocr_context.get('ocr_provider')}：段落 {len(paragraphs)} 則")

    log(f"[2] 模型 {model} @ {client.base_url}")

    # [2] 版面與文字
    structure = analyze_structure(client, page_info, det_path, paragraphs,
                                 str(ocr_context.get("ocr_provider") or ""), enhanced_path)
    (debug_dir / "structure.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[2] 標題「{structure.get('title', '')}」，文字塊 {len(structure.get('blocks') or [])} 個")

    # [3] 關鍵圖片
    visuals = analyze_visuals(client, page_info, det_path, structure, enhanced_path)
    (debug_dir / "visuals.json").write_text(
        json.dumps(visuals, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[3] 視覺元素 {len(visuals.get('visual_elements') or [])} 個")

    # [4] 裁切 ＋ 二次判讀
    images, other_artwork, warnings_ = crop_visuals(
        page_path, page_info, visuals.get("visual_elements") or [], doc_id, out_dir,
        crop_pad=options.crop_pad, snap=options.snap,
        column_ranges=(structure.get("layout") or {}).get("column_ranges") or [])
    other_artwork = [str(x) for x in visuals.get("other_artwork") or []] + other_artwork

    if options.review and images:
        images, extra_artwork, review_warnings = review_crops(client, images, out_dir,
                                                             max_images=options.max_images)
        other_artwork += extra_artwork
        warnings_ += review_warnings
    log(f"[4] 入庫圖片 {len(images)} 張（{doc_id}/），裝飾 {len(other_artwork)} 筆")

    # [5] 組裝 ＋ 輸出
    doc = build_document(
        doc_id=doc_id, page_number=options.page, source_image=image_path, page_info=page_info,
        ocr_context=ocr_context, structure=structure, visuals=visuals, images=images,
        other_artwork=list(dict.fromkeys(other_artwork)), warnings_=warnings_, model=model,
    )
    add_step(doc, "rectify", size=page_info["coordinate_space"], applied=page_info["applied"])
    add_step(doc, "ocr", provider=doc["ocr_provider"], paragraphs=len(paragraphs))
    add_step(doc, "vlm_structure", model=model, blocks=len(blocks_of(doc)))
    add_step(doc, "vlm_visuals", model=model, images=len(doc["images"]))
    add_step(doc, "crop", review=options.review, crop_pad=options.crop_pad, snap=options.snap)

    doc_path = save_document(doc, out_dir / f"{doc_id}.json")
    log(f"[5] 文件 JSON → {doc_path}")

    if options.write_md:
        from json_to_md import write_markdown

        log(f"[5] Markdown（debug）→ {write_markdown(doc, out_dir / f'{doc_id}.md')}")

    if options.write_records:
        records = to_vector_records(doc)
        log(f"[5] 向量記錄 {len(records)} 筆 → "
            f"{write_records_jsonl(records, out_dir / f'{doc_id}.records.jsonl')}")

    issues = validate_document(doc, out_dir)
    for issue in issues:
        log(f"    {issue}")
    errors = [x for x in issues if x.startswith("E")]
    log(f"[6] 驗證：{len(errors)} 錯誤 / {len(issues) - len(errors)} 警告"
        f"（人工終檢 §7 仍必做）")
    if errors and options.strict:
        raise SystemExit(2)
    return doc


def blocks_of(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc.get("blocks") or []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mannual_parser.py",
        description="說明書圖片 → 頁面文件 JSON（→ md 預覽 / 向量資料庫記錄）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例：\n"
            "  python mannual_parser.py parse assets/bbw/bbw_1.jpg --page 1\n"
            "  python mannual_parser.py parse assets/bbw/bbw_2.jpg --page 2 --ocr azure --doc-id bbw_2\n"
            "  python mannual_parser.py md output/bbw_1/bbw_1.json --check\n"
            "  python mannual_parser.py records output/bbw_1/bbw_1.json\n"
            "  python mannual_parser.py validate output/bbw_1/bbw_1.json\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--out", type=Path, default=None, help="輸出資料夾（預設 output/<檔名>）")
        target.add_argument("--ocr", choices=["paddle", "azure", "none"], default="paddle",
                            help="OCR 來源（跑在矯正後的 page.png 上，預設 paddle）")
        target.add_argument("--orientation", choices=["auto", "exif", "none", "90", "180", "270"],
                            default="auto", help="朝向矯正策略（§1.1）")
        target.add_argument("--device", default="cpu", help="PaddleOCR device")
        target.add_argument("--ocr-unwarp", action="store_true",
                            help="請 Paddle 再去彎曲／判方向（朝向已由 [0] 處理，開下去 4K 圖會多花十倍時間）")
        target.add_argument("--grid", action="store_true", help="另外產 100px 標線圖（人工定位用）")

    parse_cmd = sub.add_parser("parse", help="全流程：圖片 → JSON（＋md＋records）")
    parse_cmd.add_argument("image", type=Path)
    parse_cmd.add_argument("--page", type=int, default=None, help="頁碼，寫進 front matter")
    parse_cmd.add_argument("--doc-id", default="", help="文件 id（預設取檔名）")
    parse_cmd.add_argument("--page-context", type=Path, default=None, help="沿用既有 page_context.json")
    parse_cmd.add_argument("--provider", choices=["auto", "gpt", "ollama", "openai", "azure"],
                           default="auto",
                           help="模型後端（都走 OpenAI 相容端點）；auto＝.env 有 Azure 金鑰就 gpt")
    parse_cmd.add_argument("--det-long-side", type=int, default=0,
                           help="送給模型的偵測圖長邊（0＝看 provider：ollama 1568／gpt 2048）")
    parse_cmd.add_argument("--model", default="", help="模型名（預設 AZURE_MODEL／OLLAMA_MODEL）")
    parse_cmd.add_argument("--host", default="",
                           help="OpenAI 相容 base_url（預設 AZURE_OPENAI_ENDPOINT／OLLAMA_HOST）")
    parse_cmd.add_argument("--no-enhance", action="store_true", help="不附加高通增強圖")
    parse_cmd.add_argument("--no-review", action="store_true", help="不做逐張裁切圖二次判讀（省時間）")
    parse_cmd.add_argument("--max-images", type=int, default=8, help="二次判讀最多幾張")
    parse_cmd.add_argument("--crop-pad", type=int, default=15, help="裁切留白像素")
    parse_cmd.add_argument("--no-snap", action="store_true", help="不用墨韻投影微調 bbox")
    parse_cmd.add_argument("--no-md", action="store_true", help="不產 md（DB 流程其實不需要它）")
    parse_cmd.add_argument("--no-records", action="store_true", help="不產 records.jsonl")
    parse_cmd.add_argument("--strict", action="store_true", help="驗證有錯誤就非零 exit")
    add_common(parse_cmd)
    parse_cmd.set_defaults(func=command_parse)

    page_cmd = sub.add_parser("page", help="只做矯正＋OCR（先確認座標系與草稿）")
    page_cmd.add_argument("image", type=Path)
    add_common(page_cmd)
    page_cmd.set_defaults(func=command_page)

    md_cmd = sub.add_parser("md", help="文件 JSON → Markdown（debug 預覽，可重渲）")
    md_cmd.add_argument("document", type=Path)
    md_cmd.add_argument("-o", "--out", type=Path, default=None)
    md_cmd.add_argument("--check", action="store_true", help="順就跑驗證")
    md_cmd.set_defaults(func=command_md)

    records_cmd = sub.add_parser("records", help="文件 JSON → 向量資料庫記錄")
    records_cmd.add_argument("document", type=Path)
    records_cmd.add_argument("-o", "--out", type=Path, default=None)
    records_cmd.add_argument("--stdout", action="store_true", help="直接印出（除錯）")
    records_cmd.set_defaults(func=command_records)

    validate_cmd = sub.add_parser("validate", help="跑 §6 交付前驗證")
    validate_cmd.add_argument("document", type=Path)
    validate_cmd.add_argument("--strict", action="store_true")
    validate_cmd.set_defaults(func=command_validate)
    return parser


def command_parse(args: argparse.Namespace) -> None:
    out_dir = args.out or Path("output") / args.image.stem
    options = ParseOptions(
        page=args.page, doc_id=args.doc_id, provider=args.provider, ocr=args.ocr,
        page_context=args.page_context,
        orientation=args.orientation, model=args.model, host=args.host,
        enhance=not args.no_enhance, review=not args.no_review, max_images=args.max_images,
        crop_pad=args.crop_pad, snap=not args.no_snap, grid=args.grid,
        write_md=not args.no_md, write_records=not args.no_records,
        strict=args.strict, device=args.device, ocr_unwarp=args.ocr_unwarp,
        det_long_side=args.det_long_side,
    )
    parse_page(args.image, out_dir, options)


def command_page(args: argparse.Namespace) -> None:
    load_env()
    out_dir = args.out or Path("output") / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = safe_doc_id(args.image.stem)
    page_path = out_dir / f"{doc_id}.page.png"
    page_info = rectify_page(args.image.resolve(), page_path, args.orientation)
    print(f"[0] {page_info['width']}x{page_info['height']} → {page_path}")
    if args.grid:
        print(f"[0] 標線圖 → {draw_grid_overlay(page_path, out_dir / '_debug' / 'grid.png')}")
    context = run_ocr(page_path, out_dir / "ocr", args.ocr, args.device, args.ocr_unwarp)
    print(f"[1] OCR {context.get('ocr_provider')}：段落 {len(context.get('paragraphs') or [])} 則"
          f"／文字 {len(str(context.get('ocr_text') or ''))} 字 → {out_dir / 'ocr'}")


def command_md(args: argparse.Namespace) -> None:
    from json_to_md import write_markdown

    doc = load_document(args.document)
    out = args.out or args.document.with_suffix(".md")
    print(f"Markdown → {write_markdown(doc, out)}")
    if args.check:
        _report_validation(doc, args.document.parent)


def command_records(args: argparse.Namespace) -> None:
    doc = load_document(args.document)
    records = to_vector_records(doc)
    if args.stdout:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
        return
    out = args.out or args.document.with_name(f"{doc.get('id', 'doc')}.records.jsonl")
    print(f"{len(records)} 筆記錄 → {write_records_jsonl(records, out)}")


def command_validate(args: argparse.Namespace) -> None:
    doc = load_document(args.document)
    errors = _report_validation(doc, args.document.parent)
    if errors and args.strict:
        raise SystemExit(2)


def _report_validation(doc: dict[str, Any], base_dir: Path) -> int:
    issues = validate_document(doc, base_dir)
    for issue in issues:
        print(issue)
    errors = len([x for x in issues if x.startswith("E")])
    print(f"{'通過' if not errors else f'{errors} 個錯誤'}／{len(issues)} 個問題"
          f"（自動化產物仍需人工終檢 §7）")
    return errors


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
