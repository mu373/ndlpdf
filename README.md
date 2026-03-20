# ndlpdf

Convert scanned PDFs to searchable PDFs using [ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) OCR.

ndlocr-lite is an OCR engine for Japanese documents developed by the National Diet Library of Japan. It supports vertical text and reading-order detection. ndlpdf wraps this into a CLI that takes a PDF in and produces a searchable PDF out.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install as a tool (available globally)
uv tool install git+https://github.com/mu373/ndlpdf

# Or clone and install locally
git clone https://github.com/mu373/ndlpdf
cd ndlpdf
uv sync
```

This installs ndlocr-lite (with ~150MB of ONNX model weights) from GitHub.

## Usage

```bash
# If installed as a uv tool
ndlpdf input.pdf

# If cloned locally
uv run ndlpdf input.pdf

# Specify output path
ndlpdf input.pdf -o output.pdf

# Also extract text to a .txt file
ndlpdf input.pdf --txt

# Use GPU
ndlpdf input.pdf --device cuda

# Adjust render DPI (default: 200)
ndlpdf input.pdf --dpi 300

# Debug mode — visible red text + blue bounding boxes
ndlpdf input.pdf --debug
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output` | Output PDF path | `{stem}_ocr.pdf` |
| `--device` | `cpu` or `cuda` | `cpu` |
| `--dpi` | Render DPI for OCR | `200` |
| `--txt` | Also write a `.txt` sidecar file | off |
| `--debug` | Visible red text + blue bounding boxes | off |

## Acknowledgements

- OCR engine: [ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) (CC BY 4.0, National Diet Library, Japan)

## License

MIT
