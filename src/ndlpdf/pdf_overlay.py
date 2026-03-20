"""PDF text overlay: create and merge invisible OCR text layers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject
from reportlab.lib.colors import blue, red
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def _register_fonts() -> None:
    """Register CID fonts for Japanese text.

    All text uses horizontal-mode fonts so that pdftotext (Poppler) can
    extract the invisible layer reliably.  Vertical CID fonts (WMode=1)
    cause pdftotext to misread character positions.
    """
    registered = set(pdfmetrics.getRegisteredFontNames())
    if "HeiseiKakuGo-W5" not in registered:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5", isVertical=False))


def _draw_text_layer(c: canvas.Canvas, img_h, ocr_results, debug=False):
    """Draw OCR text onto a reportlab canvas.

    All text uses HeiseiKakuGo-W5 (horizontal mode) for pdftotext compatibility.
    Coordinates are in the displayed image space (y-down for bbox, converted
    to PDF y-up internally).

    Parameters
    ----------
    c : canvas.Canvas
        reportlab canvas to draw on.
    img_h : float
        Page height in points (for y-axis flip).
    ocr_results : list[dict]
        OCR result dicts with ``bbox``, ``text``, ``is_vertical``.
    debug : bool
        If True, draw visible red text and blue bounding boxes.
    """
    _register_fonts()
    font_name = "HeiseiKakuGo-W5"
    if debug:
        c.setFillColor(red, alpha=0.3)
        c.setStrokeColor(blue, alpha=0.5)
    else:
        c.setFillColor(blue, alpha=0.0)
    for result in ocr_results:
        bbox = result["bbox"]
        text = result["text"]
        if result["is_vertical"]:
            size = bbox[2] * 3 / 4
            x = bbox[0] + bbox[2] / 2
            y = img_h - bbox[1]
        else:
            size = bbox[3]
            x = bbox[0]
            y = img_h - (bbox[1] + bbox[3] / 2)
        if size < 1:
            continue
        if debug:
            c.rect(
                bbox[0],
                img_h - bbox[1] - bbox[3],
                bbox[2],
                bbox[3],
                fill=0,
                stroke=1,
            )
        c.setFont(font_name, size)
        c.drawString(x, y, text)


def create_text_only_pdf(
    page_w_pt,
    page_h_pt,
    ocr_results,
    scale,
    rotation=0,
    debug=False,
) -> BytesIO:
    """Create a single-page PDF containing only the OCR text layer.

    Handles page rotation by applying a canvas transform so that OCR
    pixel coordinates (from the rendered/displayed image) map correctly
    to the raw PDF mediabox coordinate space.

    Parameters
    ----------
    page_w_pt : float
        Raw mediabox width in points (before rotation).
    page_h_pt : float
        Raw mediabox height in points (before rotation).
    ocr_results : list[dict]
        OCR results with ``bbox`` in pixel coordinates.
    scale : float
        Pixels-per-point (``dpi / 72``).
    rotation : int
        Page ``/Rotate`` value (0, 90, 180, 270).
    debug : bool
        If True, render visible text and bounding boxes.

    Returns
    -------
    BytesIO
        Single-page PDF buffer.
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w_pt, page_h_pt))

    # The rendered image is in displayed (rotated) space.
    # Apply a canvas transform so we can draw in displayed coordinates
    # and have it map correctly to the raw mediabox coordinate space.
    # This handles both position and text orientation.
    if rotation == 270:
        # display (dx, dy) → raw (dy, page_h_pt - dx)
        c.transform(0, -1, 1, 0, 0, page_h_pt)
        displayed_h_pt = page_w_pt
    elif rotation == 90:
        # display (dx, dy) → raw (page_w_pt - dy, dx)
        c.transform(0, 1, -1, 0, page_w_pt, 0)
        displayed_h_pt = page_w_pt
    elif rotation == 180:
        c.transform(-1, 0, 0, -1, page_w_pt, page_h_pt)
        displayed_h_pt = page_h_pt
    else:
        displayed_h_pt = page_h_pt

    scaled_results = []
    for result in ocr_results:
        scaled = {
            "bbox": [v / scale for v in result["bbox"]],
            "text": result["text"],
            "is_vertical": result["is_vertical"],
            "confidence": result["confidence"],
        }
        scaled_results.append(scaled)

    _draw_text_layer(c, displayed_h_pt, scaled_results, debug=debug)
    c.save()
    buf.seek(0)
    return buf


def _strip_text(page):
    """Remove all text operations (BT...ET blocks) from a PDF page."""
    content = page.get("/Contents")
    if content is None:
        return
    cs = ContentStream(content, page.pdf)
    filtered = []
    in_text = False
    for operands, operator in cs.operations:
        if operator == b"BT":
            in_text = True
            continue
        if operator == b"ET":
            in_text = False
            continue
        if not in_text:
            filtered.append((operands, operator))
    cs.operations = filtered
    page[NameObject("/Contents")] = cs


def build_searchable_pdf(output_path, page_pdfs, original_pdf_path, *, strip_text=True):
    """Merge OCR text layers onto the original PDF pages.

    Parameters
    ----------
    output_path : str or Path
        Destination file path.
    page_pdfs : list[BytesIO]
        One text-only PDF per page.
    original_pdf_path : str or Path
        Path to the source PDF.
    strip_text : bool
        If True (default), strip existing text from each page before
        merging the OCR overlay.  Pass False to preserve original text
        (useful for mixed-content or already-searchable PDFs).
    """
    writer = PdfWriter()
    reader = PdfReader(str(original_pdf_path))
    if len(reader.pages) != len(page_pdfs):
        raise ValueError("Page count mismatch between source PDF and OCR overlay pages")
    for index, original_page in enumerate(reader.pages):
        if strip_text:
            _strip_text(original_page)
        page_pdfs[index].seek(0)
        overlay_page = PdfReader(page_pdfs[index]).pages[0]
        original_page.merge_page(overlay_page)
        writer.add_page(original_page)
    writer.write(str(Path(output_path)))
