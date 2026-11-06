#!/usr/bin/env python3
"""Extract OCR/layout/figure candidates from one photographed rulebook page.

Output contract: page_context.json, compatible with 03_ollama_qwen3vl_analyze.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption
from dotenv import load_dotenv


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "figure"


def region_info(regions: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for region in regions or []:
        result.append(
            {
                "page_number": getattr(region, "page_number", None),
                "polygon": list(getattr(region, "polygon", []) or []),
            }
        )
    return result


def run(image_path: Path, output_dir: Path) -> Path:
    load_dotenv()
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_API_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "請先設定 AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT 與 AZURE_API_KEY"
        )
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    # MVP input is one still image of one physical rulebook page.
    with image_path.open("rb") as stream:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=stream,
            output=[AnalyzeOutputOption.FIGURES],
            content_type="application/octet-stream",
        )

    result = poller.result()
    operation_id = poller.details["operation_id"]

    paragraphs: list[dict[str, Any]] = []
    paragraph_id_by_index: dict[int, str] = {}
    for index, paragraph in enumerate(result.paragraphs or []):
        paragraph_id = f"P{index + 1:04d}"
        paragraph_id_by_index[index] = paragraph_id
        paragraphs.append(
            {
                "id": paragraph_id,
                "text": paragraph.content or "",
                "regions": region_info(getattr(paragraph, "bounding_regions", None)),
            }
        )

    visual_regions: list[dict[str, Any]] = []
    for index, figure in enumerate(result.figures or []):
        figure_id = figure.id or f"figure_{index + 1}"
        related_paragraph_ids: list[str] = []
        for element in getattr(figure, "elements", None) or []:
            match = re.fullmatch(r"/paragraphs/(\d+)", str(element))
            if match:
                paragraph_id = paragraph_id_by_index.get(int(match.group(1)))
                if paragraph_id:
                    related_paragraph_ids.append(paragraph_id)

        crop_path: str | None = None
        if figure.id:
            crop_file = figures_dir / f"{safe_name(figure_id)}.png"
            image_stream = client.get_analyze_result_figure(
                model_id=result.model_id,
                result_id=operation_id,
                figure_id=figure.id,
            )
            with crop_file.open("wb") as output_stream:
                for chunk in image_stream:
                    output_stream.write(chunk)
            crop_path = str(crop_file.resolve())

        caption = ""
        if getattr(figure, "caption", None):
            caption = figure.caption.content or ""

        visual_regions.append(
            {
                "id": str(figure_id),
                "label": "figure",
                "caption": caption,
                "crop_path": crop_path,
                "related_paragraph_ids": sorted(set(related_paragraph_ids)),
                "regions": region_info(getattr(figure, "bounding_regions", None)),
            }
        )

    # Include tables as text context as well as ordinary paragraphs.
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(result.tables or []):
        cells = []
        for cell in table.cells or []:
            cells.append(
                {
                    "row": cell.row_index,
                    "column": cell.column_index,
                    "text": cell.content or "",
                }
            )
        tables.append(
            {
                "id": f"T{table_index + 1:04d}",
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": cells,
            }
        )

    ocr_text = result.content or "\n\n".join(p["text"] for p in paragraphs)
    context = {
        "schema_version": "1.0",
        "source_image": str(image_path.resolve()),
        "ocr_provider": "azure-document-intelligence-prebuilt-layout",
        "page_count": len(result.pages or []),
        "ocr_text": ocr_text,
        "paragraphs": paragraphs,
        "tables": tables,
        "visual_regions": visual_regions,
    }

    context_path = output_dir / "page_context.json"
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Keep the full service response for troubleshooting/auditing.
    raw_path = output_dir / "azure_layout_raw.json"
    raw_path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"OCR paragraphs: {len(paragraphs)}")
    print(f"Tables: {len(tables)}")
    print(f"Figures extracted: {len(visual_regions)}")
    print(f"Context: {context_path}")
    return context_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用 Azure Document Intelligence 解析一張桌遊規則書頁面"
    )
    parser.add_argument("image", type=Path, help="單頁照片，例如 page_001.jpg")
    parser.add_argument("--out", type=Path, default=Path("output_azure"))
    args = parser.parse_args()
    run(args.image, args.out)


if __name__ == "__main__":
    main()
