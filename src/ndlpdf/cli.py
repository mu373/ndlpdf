"""CLI entry point for ndlpdf: convert scanned PDFs to searchable PDFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pypdfium2
from pypdf import PdfReader
from tqdm import tqdm

from ndlpdf.ocr_engine import load_models, ocr_page
from ndlpdf.pdf_overlay import build_searchable_pdf, create_text_only_pdf

_MIN_DPI = 50
_MAX_DPI = 1200


def main():
    """CLI entry point: OCR a PDF and produce a searchable copy.

    Renders each page with pypdfium2, runs ndlocr-lite OCR, and merges
    an invisible text layer onto the original PDF.
    """
    parser = argparse.ArgumentParser(
        description="Convert scanned PDF to searchable PDF using ndlocr-lite OCR",
    )
    parser.add_argument("input", help="Path to the input PDF")
    parser.add_argument("-o", "--output", help="Path to the output PDF")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for OCR inference",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help=f"Rasterization DPI used during OCR ({_MIN_DPI}–{_MAX_DPI})",
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help="Write extracted text to a sidecar .txt file",
    )
    parser.add_argument(
        "--no-strip-text",
        action="store_true",
        help="Keep existing text in the original PDF (default strips it)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show visible red text and bounding boxes for the OCR layer",
    )
    args = parser.parse_args()

    if args.dpi < _MIN_DPI or args.dpi > _MAX_DPI:
        parser.error(f"--dpi must be between {_MIN_DPI} and {_MAX_DPI}, got {args.dpi}")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input PDF does not exist: {input_path}")
    if not input_path.is_file():
        parser.error(f"Input path is not a file: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_ocr.pdf")
    )

    doc = pypdfium2.PdfDocument(str(input_path))
    pdf_reader = PdfReader(str(input_path))
    detector, rec30, rec50, rec100 = load_models(args.device)

    page_pdfs = []
    all_text = []
    failed_pages = []
    scale = args.dpi / 72

    try:
        for i in tqdm(range(len(doc)), desc="OCR", unit="page"):
            bitmap = None
            page = None
            try:
                page = doc[i]
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil().convert("RGB")
                np_image = np.array(pil_image)
                del pil_image

                ocr_results = ocr_page(
                    np_image,
                    f"page_{i}",
                    detector,
                    rec30,
                    rec50,
                    rec100,
                )
                del np_image
                all_text.append("\n".join(result["text"] for result in ocr_results))

                # Use raw mediabox dimensions (before rotation) for the overlay
                pypdf_page = pdf_reader.pages[i]
                mb = pypdf_page.mediabox
                raw_w = float(mb.width)
                raw_h = float(mb.height)
                rotation = int(pypdf_page.get("/Rotate") or 0) % 360

                page_pdfs.append(
                    create_text_only_pdf(
                        raw_w,
                        raw_h,
                        ocr_results,
                        scale,
                        rotation=rotation,
                        debug=args.debug,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failed_pages.append((i, exc))
                print(f"\nWarning: page {i + 1} failed: {exc}", file=sys.stderr)
                all_text.append("")
                # Blank overlay so page count stays in sync
                pypdf_page = pdf_reader.pages[i]
                mb = pypdf_page.mediabox
                page_pdfs.append(
                    create_text_only_pdf(
                        float(mb.width), float(mb.height), [], scale,
                    )
                )
            finally:
                if bitmap is not None and hasattr(bitmap, "close"):
                    bitmap.close()
                if page is not None and hasattr(page, "close"):
                    page.close()
    finally:
        if hasattr(doc, "close"):
            doc.close()

    build_searchable_pdf(
        output_path, page_pdfs, input_path, strip_text=not args.no_strip_text,
    )

    txt_path = None
    if args.txt:
        txt_path = output_path.with_suffix(".txt")
        txt_path.write_text("\n\n".join(all_text), encoding="utf-8")

    print(f"Created searchable PDF: {output_path}")
    if txt_path is not None:
        print(f"Wrote extracted text: {txt_path}")
    n_ok = len(page_pdfs) - len(failed_pages)
    print(f"Processed {n_ok}/{len(page_pdfs)} pages at {args.dpi} DPI")
    if failed_pages:
        print(
            f"Warning: {len(failed_pages)} page(s) failed OCR: "
            f"{', '.join(str(p + 1) for p, _ in failed_pages)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
