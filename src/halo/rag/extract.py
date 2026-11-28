"""Richer document extraction: PDF tables and OCR for scanned pages and images.

* **Tables** (pdfplumber): each table on a PDF page becomes its own chunk, written as a
  Markdown table, so rows and columns survive chunking and the model can read them.
* **OCR** (Tesseract): PDF pages with almost no extractable text (scans, photos of pages)
  are rendered and read with Tesseract, and image files (.png, .jpg, .tiff, ...) can be
  ingested directly. Needs the `tesseract` binary (`apt install tesseract-ocr`,
  `brew install tesseract`); without it, OCR is skipped with a warning.
"""
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Dict, List, Optional

from halo.core.config import settings
from halo.core.log import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


@lru_cache(maxsize=1)
def tesseract_path() -> Optional[str]:
    if (settings.OCR or "auto").lower() == "off":
        return None
    return shutil.which(settings.TESSERACT_CMD or "tesseract")


def ocr_available() -> bool:
    return tesseract_path() is not None


def ocr_image(path: str) -> str:
    """Text from an image file via the Tesseract CLI (no Python wrapper needed)."""
    cmd = tesseract_path()
    if not cmd:
        raise RuntimeError("OCR needs Tesseract: install it (apt install tesseract-ocr / brew install tesseract) "
                           "or set OCR=off")
    result = subprocess.run([cmd, path, "stdout", "-l", settings.OCR_LANG, "--psm", "3"],
                            capture_output=True, text=True, timeout=settings.OCR_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(f"Tesseract failed: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def ocr_pdf_page(path: str, page_number: int) -> str:
    """Render one PDF page (1-based) at 300 dpi and OCR it."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        image = pdf.pages[page_number - 1].to_image(resolution=300)
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "page.png")
            image.save(png)
            return ocr_image(png)


def _cell(value) -> str:
    return " ".join(str(value).split()).replace("|", "\\|") if value is not None else ""


def table_to_markdown(rows: List[List]) -> str:
    """Rows from pdfplumber -> a Markdown table. The first non-empty row is the header."""
    rows = [[_cell(c) for c in row] for row in rows if row and any(c not in (None, "") for c in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if width < 2 and len(rows) < 2:
        return ""
    head, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def extract_pdf_tables(path: str) -> Dict[int, List[str]]:
    """{page number (1-based): [markdown table, ...]} for every table pdfplumber finds."""
    if not settings.EXTRACT_TABLES:
        return {}
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; PDF tables are not extracted")
        return {}
    tables: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                for rows in page.extract_tables():
                    md = table_to_markdown(rows)
                    if md:
                        tables.setdefault(number, []).append(md)
    except Exception:
        logger.exception("Table extraction failed for %s; continuing with text only", path)
    return tables
