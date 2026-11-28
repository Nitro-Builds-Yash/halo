"""Real-file tests for table extraction and OCR (OCR tests need the tesseract binary)."""
import shutil

import pytest

from halo.rag import extract, ingest

needs_ocr = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
FONT_TEXT = "Refunds are accepted within 30 days of purchase."


def _font(size=40):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_image(path, lines):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1700, 140 + 80 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((80, 70 + 80 * i), line, fill="black", font=_font())
    img.save(path)
    return img


def test_table_to_markdown():
    md = extract.table_to_markdown([["Plan", "Price", None], ["Basic", "$5", ""], [None, None, None], ["Pro|Max", "$9\nmonthly", "x"]])
    assert md.splitlines() == ["| Plan | Price |  |", "|---|---|---|", "| Basic | $5 |  |", "| Pro\\|Max | $9 monthly | x |"]
    assert extract.table_to_markdown([[None, ""]]) == ""


def test_pdf_tables_become_chunks(tmp_path):
    rl = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

    pdf = tmp_path / "pricing.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4)
    table = Table([["Plan", "Monthly price", "Seats"], ["Basic", "$5", "1"], ["Team", "$40", "10"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    doc.build([Paragraph("Our pricing plans are listed below.", getSampleStyleSheet()["Normal"]), table])

    docs = ingest.load_file_with_metadata(str(pdf))
    tables = [d for d in docs if d.metadata.get("content_type") == "table"]
    assert len(tables) == 1 and tables[0].metadata["page"] == 1
    assert "| Plan | Monthly price | Seats |" in tables[0].page_content
    assert "| Team | $40 | 10 |" in tables[0].page_content
    assert any("pricing plans" in d.page_content for d in docs if "content_type" not in d.metadata)


@needs_ocr
def test_ocr_image_file(tmp_path):
    png = tmp_path / "notice.png"
    text_image(png, [FONT_TEXT])
    docs = ingest.load_file_with_metadata(str(png))
    assert docs[0].metadata == {"source": "notice.png", "page": 1, "ocr": True}
    assert "30 days" in docs[0].page_content and "Refunds" in docs[0].page_content


@needs_ocr
def test_scanned_pdf_pages_are_ocrd(tmp_path):
    img = text_image(tmp_path / "scan.png", [FONT_TEXT, "Receipts are required for every refund."])
    pdf = tmp_path / "scanned.pdf"
    img.save(pdf)  # an image-only PDF: no text layer, like a scanner produces
    docs = ingest.load_file_with_metadata(str(pdf))
    assert docs and docs[0].metadata["ocr"] is True and docs[0].metadata["page"] == 1
    assert "30 days" in docs[0].page_content


def test_scanned_pdf_without_ocr_is_skipped(tmp_path, monkeypatch):
    img = text_image(tmp_path / "scan.png", [FONT_TEXT])
    pdf = tmp_path / "scanned.pdf"
    img.save(pdf)
    monkeypatch.setattr(extract.settings, "OCR", "off")
    extract.tesseract_path.cache_clear()
    try:
        assert ingest.load_file_with_metadata(str(pdf)) == []
        with pytest.raises(RuntimeError, match="Tesseract"):
            ingest.load_file_with_metadata(str(tmp_path / "scan.png"))
    finally:
        extract.tesseract_path.cache_clear()
