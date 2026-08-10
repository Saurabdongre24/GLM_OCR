#!/usr/bin/env python
"""GLM-OCR invoice extractor — bounding boxes + text, three staged passes.

Standalone: does not import anything from the ADE backend. Same logic as
``notebooks/glm_ocr_invoice_test.ipynb``, restructured as a CLI.

    Stage 1  PP-DocLayoutV3   -> page regions (bbox + label)
    Stage 2  GLM-OCR          -> text for each region       [batched]
    Stage 3  OpenCV / numpy   -> per-cell bbox inside tables, + cell text [batched]

GLM-OCR itself returns no coordinates at any granularity — it is an image->text
model. Every bbox comes from stage 1 (regions) or stage 3 (cells).

Usage
-----
    python scripts/glm_ocr_invoice.py invoice.pdf
    python scripts/glm_ocr_invoice.py invoice.pdf --out results --dpi 150
    python scripts/glm_ocr_invoice.py invoice.pdf --pages 0,1 --batch-size 12
    python scripts/glm_ocr_invoice.py invoice.pdf --cell-text none --no-overlay
    python scripts/glm_ocr_invoice.py table.png --image      # a cropped table image

Outputs (under --out)
---------------------
    page_000.png              rendered page
    page_000_overlay.png      regions (thick) + table cells (thin red)
    regions.json              every region: label, bbox, text, cells[]
    fullpage.json             whole-page OCR baseline (only with --fullpage)
    summary.txt               the console report

Requires: transformers, torch, pypdfium2, pillow, opencv-python, numpy, timm.
First run downloads ~2 GB (GLM-OCR) + ~200 MB (PP-DocLayoutV3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# The Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used in progress output — printing one raises UnicodeEncodeError and
# kills the run. Force UTF-8 and degrade gracefully if a glyph is unmappable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already wrapped, or not a TTY
        pass

# torch / transformers are imported lazily inside the model wrappers so that the
# pure-geometry helpers below can be imported and tested without a GPU stack.

# ===========================================================================
#  >>>>>  EDIT HERE  <<<<<   set your PDF path, then just run this file.
#  Any command-line argument overrides these.
# ===========================================================================

PDF_PATH = r"D:\ADE_Code_Path\doclayoutdetection\samples\invoice.pdf"

OUT_DIR = r"glm_ocr_out"   # where results are written
DPI = 175                  # 150 = fast, 200 = balanced, 300 = small fonts
PAGES = None               # None = all pages, or [0] / [0, 1] for specific ones
BATCH_SIZE = 12            # crops per GPU call — drop to 6 if you hit CUDA OOM
CELL_TEXT = "derive"       # "derive" = cell text from the table HTML, no extra
                           #            model calls (fastest — recommended)
                           # "batch"  = OCR every cell (most accurate, slowest)
                           # "none"   = cell boxes only, no text

# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GLM_MODEL_ID = "zai-org/GLM-OCR"
LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"

# Cell-detection tunables — sized for document-scale tables at 150–300 DPI.
BINARIZE_THRESHOLD = 185
H_LINE_KERNEL_PX = 50      # min run length to count as a horizontal rule
V_LINE_KERNEL_PX = 25      # min run length to count as a vertical rule
CLOSE_KERNEL_PX = 4
CLOSE_ITERATIONS = 2
MIN_CELL_W_PX = 20
MIN_CELL_H_PX = 10
MIN_CELL_AREA_PX = 500
MAX_CELL_AREA_FRAC = 0.95  # reject a contour covering nearly the whole crop
CONTAINMENT_THRESH = 0.90

# Per-label generation budget. A cell holding "H" does not need 2048 tokens.
MAX_TOK = {
    "table": 1536, "cell": 48, "header": 128, "title": 64,
    "footer": 128, "key_value": 128, "text": 512,
}
MAX_TOK_DEFAULT = 512
MAX_TOK_FULLPAGE = 2048

# PP-DocLayoutV3 raw label -> normalized region type.
LABEL_MAP = {
    "abstract": "text", "algorithm": "code", "aside_text": "text", "chart": "figure",
    "content": "text", "formula": "formula", "doc_title": "title",
    "figure_title": "caption", "footer": "footer", "footnote": "footnote",
    "formula_number": "text", "header": "header", "image": "figure",
    "number": "text", "paragraph_title": "header", "reference": "text",
    "reference_content": "text", "seal": "figure", "table": "table", "text": "text",
    "vision_footnote": "footnote", "list": "list", "code": "code", "form": "form",
    "key_value": "key_value", "stamp": "figure", "logo": "figure",
    "equation": "formula", "doc_index": "document_index",
}

REGION_COLORS = {
    "table": (220, 50, 50), "header": (50, 120, 220), "title": (140, 60, 200),
    "figure": (240, 150, 40), "footer": (110, 110, 110), "key_value": (30, 160, 120),
    "formula": (200, 120, 180), "list": (80, 170, 60), "caption": (0, 150, 180),
}


def region_prompt(label: str) -> str:
    """GLM-OCR is trained on keyword triggers that switch its internal mode.

    Short triggers beat long instructions on a 0.9B model.
    """
    if label == "table":
        return "Table Recognition:"
    if label == "formula":
        return "Formula Recognition:"
    return "Text Recognition:"


def tok_budget(label: str) -> int:
    return MAX_TOK.get(label, MAX_TOK_DEFAULT)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Timings:
    """Wall-clock per phase, for the summary table at the end of a run."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self.notes: dict[str, str] = {}
        self._t0 = time.perf_counter()

    def record(self, name: str, seconds: float, note: str = "") -> None:
        self.phases[name] = self.phases.get(name, 0.0) + seconds
        if note:
            self.notes[name] = note

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def report(self, n_pages: int) -> str:
        total = self.elapsed()
        setup = self.phases.get("load models", 0.0) + self.phases.get("render PDF", 0.0)
        width = max((len(k) for k in self.phases), default=10)

        lines = ["=" * 62, "TIMING", "=" * 62]
        for name, secs in self.phases.items():
            pct = 100 * secs / total if total else 0
            note = self.notes.get(name, "")
            lines.append(f"  {name:<{width}}  {secs:>7.1f}s  {pct:>4.0f}%  {note}")
        lines.append("-" * 62)
        lines.append(f"  {'TOTAL':<{width}}  {total:>7.1f}s         "
                     f"{n_pages} page(s) = {total / max(1, n_pages):.1f}s/page")
        if setup:
            after = total - setup
            lines.append(f"  {'per-page work':<{width}}  {after:>7.1f}s         "
                         f"excluding one-time setup = "
                         f"{after / max(1, n_pages):.1f}s/page")
        lines.append("=" * 62)
        return "\n".join(lines)


@dataclass
class Region:
    """One layout region, plus its OCR text and (for tables) its cells."""

    page_index: int
    seq_index: int
    label: str                  # normalized: text / table / header / title / ...
    raw_label: str              # what PP-DocLayoutV3 actually emitted
    layout_confidence: float
    bbox_norm: list             # [left, top, right, bottom], 0..1, origin top-left
    bbox_px: list               # same box in page pixels at the chosen DPI
    width_px: int
    height_px: int
    text: str = ""
    ocr_ms: float = 0.0
    skipped: str | None = None
    cells: list = field(default_factory=list)
    cell_html: str = ""


# ---------------------------------------------------------------------------
# Stage 0 — PDF to page images
# ---------------------------------------------------------------------------

def render_pdf(pdf_path: Path, out_dir: Path, dpi: int,
               pages: list[int] | None) -> list[Path]:
    """Rasterize the PDF. Everything downstream is vision, so we work from pixels."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    indices = pages if pages is not None else range(len(doc))
    paths: list[Path] = []
    for i in indices:
        if i >= len(doc):
            print(f"  page {i} out of range (document has {len(doc)}) — skipping")
            continue
        img = doc[i].render(scale=dpi / 72.0).to_pil().convert("RGB")
        dest = out_dir / f"page_{i:03d}.png"
        img.save(dest)
        img.close()
        paths.append(dest)
    doc.close()
    print(f"rendered {len(paths)} page(s) @ {dpi} dpi -> {out_dir}")
    return paths


# ---------------------------------------------------------------------------
# Stage 1 — layout detection
# ---------------------------------------------------------------------------

class LayoutDetector:
    """PP-DocLayoutV3 — finds page regions. Produces boxes, never text."""

    def __init__(self, model_dir: str = LAYOUT_MODEL_ID, device: str = "auto",
                 threshold: float = 0.3, batch_size: int = 4) -> None:
        self.model_dir = model_dir
        self.threshold = threshold
        self.batch_size = batch_size
        self._device_str = device
        self.model = None
        self.processor = None
        self.device = None

    def start(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        t0 = time.perf_counter()
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self._device_str == "auto" else self._device_str
        )
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_dir, trust_remote_code=True)
        self.model.to(self.device).eval()
        print(f"PP-DocLayoutV3 ready on {self.device} "
              f"({time.perf_counter() - t0:.1f}s)")

    def detect(self, images: list[Image.Image]) -> dict[int, list[Region]]:
        """-> {page_index: [Region, ...]}, sorted into reading order."""
        import torch

        id2label = self.model.config.id2label

        # Batched: one forward pass per `batch_size` pages, not per page.
        all_dets = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            all_dets.extend(self.processor.post_process_object_detection(
                out,
                target_sizes=[(im.size[1], im.size[0]) for im in batch],
                threshold=self.threshold,
            ))

        results: dict[int, list[Region]] = {}
        for page_idx, (img, det) in enumerate(zip(images, all_dets)):
            W, H = img.size
            regions: list[Region] = []
            for score, label_id, box in zip(det["scores"].cpu(),
                                            det["labels"].cpu(),
                                            det["boxes"].cpu()):
                raw = id2label.get(label_id.item(), "unknown")
                x1, y1, x2, y2 = box.tolist()
                # Normalize to 0..1 so boxes are DPI-independent, then clamp.
                norm = [min(max(v, 0.0), 1.0)
                        for v in (x1 / W, y1 / H, x2 / W, y2 / H)]
                if norm[2] <= norm[0] or norm[3] <= norm[1]:
                    continue
                px = [int(norm[0] * W), int(norm[1] * H),
                      int(norm[2] * W), int(norm[3] * H)]
                regions.append(Region(
                    page_index=page_idx, seq_index=0,
                    label=LABEL_MAP.get(raw.lower(), "text"), raw_label=raw,
                    layout_confidence=round(score.item(), 4),
                    bbox_norm=[round(v, 5) for v in norm], bbox_px=px,
                    width_px=px[2] - px[0], height_px=px[3] - px[1],
                ))
            regions.sort(key=lambda r: r.bbox_norm[1])   # top -> down
            for i, r in enumerate(regions):
                r.seq_index = i
            results[page_idx] = regions
        return results

    def stop(self) -> None:
        import torch
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2 — GLM-OCR, batched
# ---------------------------------------------------------------------------

class GlmOCR:
    """GLM-OCR wrapper. Batches crops — a batch of 1 leaves the GPU mostly idle."""

    def __init__(self, model_id: str = GLM_MODEL_ID, batch_size: int = 8,
                 max_crop_px: int = 1600, device_map: str = "auto") -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_crop_px = max_crop_px
        self._device_map = device_map
        self.model = None
        self.processor = None

    def start(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import GlmOcrForConditionalGeneration as ModelClass
            which = "GlmOcrForConditionalGeneration"
        except ImportError:                       # older transformers
            from transformers import AutoModelForImageTextToText as ModelClass
            which = "AutoModelForImageTextToText"

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Decoder-only batched generation needs LEFT padding, otherwise short
        # samples generate from the middle of their pad run and come back empty.
        try:
            self.processor.tokenizer.padding_side = "left"
        except AttributeError:
            pass

        kwargs = dict(device_map=self._device_map, attn_implementation="sdpa")
        try:
            self.model = ModelClass.from_pretrained(
                self.model_id, dtype=torch.float16, **kwargs)
        except TypeError:                         # older transformers
            self.model = ModelClass.from_pretrained(
                self.model_id, torch_dtype=torch.float16, **kwargs)
        self.model.eval()
        print(f"GLM-OCR ready via {which} on {self.model.device} "
              f"({time.perf_counter() - t0:.1f}s)")

    def _fit(self, img: Image.Image) -> Image.Image:
        """Vision tokens scale with pixel area — cap the longest side."""
        w, h = img.size
        longest = max(w, h)
        if longest <= self.max_crop_px:
            return img
        s = self.max_crop_px / longest
        return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)

    def _prepare(self, idxs: list[int], jobs: list[tuple]):
        """CPU-side batch prep: resize crops + tokenize. Runs off the hot path."""
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": self._fit(jobs[i][0])},
            {"type": "text", "text": jobs[i][1]},
        ]}] for i in idxs]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        )

    def run(self, jobs: list[tuple], desc: str = "ocr") -> list[str]:
        """jobs: [(image, prompt, max_new_tokens), ...] -> [text, ...] in order.

        Batches are grouped by TOKEN BUDGET first, then by image area.

        Budget grouping matters more than area. ``generate()`` runs until every
        sequence in the batch emits EOS, so one 1536-token table batched with
        seven 128-token headers holds all eight hostage for the table's whole
        generation. Same-budget crops finish together.

        Area is the secondary key: it keeps the vision-token padding even.
        """
        import torch

        if not jobs:
            return []

        # Group by budget, then area within each group. Concatenating the groups
        # means a batch only spans two budgets at its boundary, not all of them.
        order = sorted(
            range(len(jobs)),
            key=lambda i: (jobs[i][2], jobs[i][0].size[0] * jobs[i][0].size[1]),
        )
        out = [""] * len(jobs)
        t0 = time.perf_counter()
        gen_tokens = 0

        batches = [order[s:s + self.batch_size]
                   for s in range(0, len(order), self.batch_size)]

        # CPU/GPU overlap: resizing crops and tokenizing is CPU work that would
        # otherwise leave the GPU idle between batches. One worker thread
        # prepares batch N+1 while the GPU generates batch N. PIL and the
        # tokenizer release the GIL, so this really does run concurrently.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self._prepare, batches[0], jobs)

            for bi, idxs in enumerate(batches):
                inputs = pending.result()
                if bi + 1 < len(batches):
                    pending = pool.submit(self._prepare, batches[bi + 1], jobs)
                inputs = inputs.to(self.model.device)

                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max(jobs[i][2] for i in idxs),
                        do_sample=False,
                    )

                # Drop the prompt tokens; keep only what the model produced.
                new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                gen_tokens += int(new_tokens.shape[0] * new_tokens.shape[1])
                decoded = self.processor.batch_decode(new_tokens,
                                                      skip_special_tokens=True)
                for i, text in zip(idxs, decoded):
                    out[i] = _strip_fences(text)

                done = min((bi + 1) * self.batch_size, len(order))
                spent = time.perf_counter() - t0
                eta = spent / done * (len(order) - done)
                print(f"  {desc}: {done}/{len(order)}  {spent:.0f}s"
                      f"  eta {eta:.0f}s   ", end="\r", flush=True)

        elapsed = time.perf_counter() - t0
        # tok/s is the number to watch: generation is autoregressive, so total
        # time tracks tokens produced far more closely than crops processed.
        print(f"  {desc}: {len(order)} crops in {elapsed:.1f}s "
              f"({elapsed / len(order):.2f}s each, "
              f"{gen_tokens / max(elapsed, 1e-6):.0f} tok/s)" + " " * 8)
        return out


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 3 — table cells
# ---------------------------------------------------------------------------

def detect_cells(crop_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bordered tables: cells are the holes between the ruling lines.

    binarize -> keep only long H/V runs (erases all text, leaves the grid) ->
    OR -> dilate to close gaps -> invert -> every enclosed contour is one cell.
    Returns crop-local pixel boxes. Empty list for borderless tables.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    _, binary = cv2.threshold(gray, BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (H_LINE_KERNEL_PX, 1)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, V_LINE_KERNEL_PX)))
    mask = cv2.bitwise_or(h_lines, v_lines)
    mask = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX)),
        iterations=CLOSE_ITERATIONS)

    contours, hierarchy = cv2.findContours(
        cv2.bitwise_not(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []

    crop_area = h * w
    boxes: list[tuple[int, int, int, int]] = []
    for i, contour in enumerate(contours):
        # hierarchy[0][i] = [next, prev, first_child, parent]. A cell is a LEAF;
        # anything with children is a container (the table outline, or the page
        # background). Without this test the outline survives the area cap and
        # the containment dedup then deletes every real cell in its favour.
        if int(hierarchy[0][i][2]) != -1:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if cw < MIN_CELL_W_PX or ch < MIN_CELL_H_PX:
            continue
        if area < MIN_CELL_AREA_PX or area > crop_area * MAX_CELL_AREA_FRAC:
            continue
        boxes.append((x, y, x + cw, y + ch))

    # Drop any box >=90% inside a larger kept box. Largest first, so a survivor
    # is never deleted by something it contains.
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                      reverse=True):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if any(_containment(box, k) >= CONTAINMENT_THRESH for k in kept):
            continue
        if area > 0:
            kept.append(box)

    kept.sort(key=lambda b: (b[1], b[0]))     # top -> down, left -> right
    return kept


def _containment(inner: tuple, outer: tuple) -> float:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix * iy) / area if area > 0 else 0.0


def detect_cells_borderless(crop_bgr: np.ndarray, min_gap_frac: float = 0.010,
                            ink_frac: float = 0.01) -> list[tuple]:
    """Borderless tables: columns are the whitespace gutters, rows the blank bands.

    Model-free projection profiles. Cannot infer merged cells — every
    row x column intersection becomes its own cell.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    _, ink_img = cv2.threshold(gray, BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    ink = (ink_img > 0).astype(np.uint8)

    def bands(profile, min_gap, limit):
        """Runs of content separated by gaps of at least `min_gap` pixels."""
        threshold = max(1, int(profile.max() * ink_frac))
        filled = profile > threshold
        out, start, gap = [], None, 0
        for i, is_filled in enumerate(filled):
            if is_filled:
                if start is None:
                    start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= min_gap:
                    out.append((start, i - gap + 1))
                    start, gap = None, 0
        if start is not None:
            out.append((start, limit))
        return out

    cols = bands(ink.sum(axis=0), max(3, int(w * min_gap_frac)), w)
    rows = bands(ink.sum(axis=1), max(2, int(h * min_gap_frac)), h)
    if len(cols) < 2 or len(rows) < 2:
        return []

    def expand(bs, limit):
        """Grow each band to the midpoint of its gutter so cells tile."""
        out = []
        for i, (s, e) in enumerate(bs):
            lo = 0 if i == 0 else (bs[i - 1][1] + s) // 2
            hi = limit if i == len(bs) - 1 else (e + bs[i + 1][0]) // 2
            out.append((lo, hi))
        return out

    cols, rows = expand(cols, w), expand(rows, h)
    return [(cx0, ry0, cx1, ry1) for (ry0, ry1) in rows for (cx0, cx1) in cols]


def assign_grid(boxes: list[tuple], extent: tuple[int, int],
                tol_frac: float = 0.02) -> list[dict]:
    """Cluster cell edges into grid lines -> row/col indices + merge spans.

    Merged cells need no special handling: a merged cell is one wider box, so
    its right edge simply lands on a further grid line and colspan follows.

    NOTE: row/col are grid-LINE indices, so they may step by 2 (0, 2, 4, ...)
    when a cell's bottom edge doesn't cluster with the next cell's top edge.
    Row grouping and spans stay exact — only the numbering is sparse.
    """
    if not boxes:
        return []
    W, H = extent

    def lines(values, tol):
        out = []
        for v in sorted(values):
            if not out or v - out[-1] > tol:
                out.append(v)
            else:
                out[-1] = (out[-1] + v) / 2     # merge near-duplicate edges
        return out

    xs = lines([b[0] for b in boxes] + [b[2] for b in boxes], W * tol_frac)
    ys = lines([b[1] for b in boxes] + [b[3] for b in boxes], H * tol_frac)

    def nearest(arr, v):
        return min(range(len(arr)), key=lambda i: abs(arr[i] - v))

    cells = []
    for (x1, y1, x2, y2) in boxes:
        c0, c1 = nearest(xs, x1), nearest(xs, x2)
        r0, r1 = nearest(ys, y1), nearest(ys, y2)
        cells.append({
            "bbox_px": [x1, y1, x2, y2],       # crop-local
            "row": r0, "col": c0,
            "rowspan": max(1, r1 - r0), "colspan": max(1, c1 - c0),
        })
    cells.sort(key=lambda c: (c["row"], c["col"]))
    return cells


def cells_to_html(cells: list[dict]) -> str:
    """Rebuild HTML from geometry, with real colspan/rowspan attributes."""
    html, current_row = ["<table>"], None
    for c in cells:
        if c["row"] != current_row:
            if current_row is not None:
                html.append("  </tr>")
            html.append("  <tr>")
            current_row = c["row"]
        attrs = ""
        if c["colspan"] > 1:
            attrs += f' colspan="{c["colspan"]}"'
        if c["rowspan"] > 1:
            attrs += f' rowspan="{c["rowspan"]}"'
        text = (c.get("text") or "").replace("\n", " ").strip()
        html.append(f"    <td{attrs}>{text}</td>")
    if current_row is not None:
        html.append("  </tr>")
    html.append("</table>")
    return "\n".join(html)


def derive_cell_text(region: Region) -> int:
    """Fill cell text from the table-level OCR — ZERO extra model calls.

    The ``Table Recognition:`` pass already read the whole table and returned it
    as HTML. Re-reading each cell individually asks the model for text it has
    already produced, which is what makes stage 3 dominate the runtime.

    Both sources are in reading order, so we match them positionally: HTML row
    *i* cell *j* -> morphology row *i* cell *j*. Returns cells filled.

    Trade-off: this trusts the model's row/column split. When the two disagree
    (model dropped a column, say) the mapping shifts. Use ``--cell-text batch``
    when per-cell accuracy matters more than speed.
    """
    import re

    if not region.cells or not region.text:
        return 0

    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", region.text,
                           re.DOTALL | re.IGNORECASE)
    if not rows_html:
        return 0

    parsed: list[list[str]] = []
    for row in rows_html:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row,
                           re.DOTALL | re.IGNORECASE)
        parsed.append([re.sub(r"<[^>]+>", " ", c).strip() for c in cells])

    # Morphology rows use grid-LINE indices, which can be sparse (0, 2, 4...).
    # Group by the row key and enumerate to get dense 0..n-1 row numbers.
    by_row: dict[int, list[dict]] = {}
    for c in region.cells:
        by_row.setdefault(c["row"], []).append(c)

    filled = 0
    for row_i, row_key in enumerate(sorted(by_row)):
        if row_i >= len(parsed):
            break
        row_cells = sorted(by_row[row_key], key=lambda c: c["col"])
        for col_i, cell in enumerate(row_cells):
            if col_i < len(parsed[row_i]):
                cell["text"] = parsed[row_i][col_i]
                cell["text_source"] = "derived"
                filled += 1
    return filled


def find_cells(region: Region, table_crop: Image.Image, page_size: tuple,
               borderless_fallback: bool = True) -> str | None:
    """Fill region.cells with boxes in BOTH coordinate spaces. Text comes later.

    Returns the method used, or None if no cells were found.
    """
    crop_bgr = cv2.cvtColor(np.array(table_crop), cv2.COLOR_RGB2BGR)

    boxes, method = detect_cells(crop_bgr), "morphology"
    if not boxes and borderless_fallback:
        boxes, method = detect_cells_borderless(crop_bgr), "projection"
    if not boxes:
        return None

    grid = assign_grid(boxes, extent=table_crop.size)
    ox, oy = region.bbox_px[0], region.bbox_px[1]
    page_w, page_h = page_size
    for c in grid:
        x1, y1, x2, y2 = c["bbox_px"]
        # Page-normalized, same convention as region.bbox_norm — this is the
        # field to store as grounding alongside region boxes.
        c["bbox_norm_page"] = [
            round((ox + x1) / page_w, 5), round((oy + y1) / page_h, 5),
            round((ox + x2) / page_w, 5), round((oy + y2) / page_h, 5),
        ]
        c["page_index"] = region.page_index
        c["method"] = method

    region.cells = grid
    return method


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(page_paths: list[Path], detector: LayoutDetector, glm: GlmOCR,
                 *, min_px: int = 20, skip_labels: set[str] | None = None,
                 detect_table_cells: bool = True, cell_text: str = "batch",
                 borderless_fallback: bool = True,
                 timings: Timings | None = None) -> list[Region]:
    """Three batched passes over the WHOLE document, not a loop of single calls.

    Batching across pages (not just within one) keeps the final batch full.
    """
    skip_labels = skip_labels or set()
    t_all = time.perf_counter()
    images = [Image.open(p).convert("RGB") for p in page_paths]

    # --- pass 1: layout ----------------------------------------------------
    t0 = time.perf_counter()
    per_page = detector.detect(images)
    n_regions = sum(len(v) for v in per_page.values())
    layout_s = time.perf_counter() - t0
    print(f"[1/3] layout    : {n_regions} regions on {len(images)} page(s) "
          f"in {layout_s:.1f}s")
    if timings:
        timings.record("stage 1 layout", layout_s, f"{n_regions} regions")

    # --- pass 2: region OCR, batched across all pages -----------------------
    all_regions: list[Region] = []
    jobs, targets = [], []
    for page_idx, regions in sorted(per_page.items()):
        page_img = images[page_idx]
        for r in regions:
            all_regions.append(r)
            if r.width_px < min_px or r.height_px < min_px:
                r.skipped = f"too small ({r.width_px}x{r.height_px}px)"
                continue
            if r.label in skip_labels:
                r.skipped = f"label '{r.label}' skipped"
                continue
            jobs.append((page_img.crop(tuple(r.bbox_px)),
                         region_prompt(r.label), tok_budget(r.label)))
            targets.append(r)

    t0 = time.perf_counter()
    print(f"[2/3] region OCR: {len(jobs)} crops, batch={glm.batch_size}")
    for region, text in zip(targets, glm.run(jobs, "  regions")):
        region.text = text
        if not text:
            region.skipped = "empty OCR response"
    ocr_s = time.perf_counter() - t0
    if timings:
        timings.record("stage 2 region OCR", ocr_s, f"{len(jobs)} crops")
    for job in jobs:
        job[0].close()

    # --- pass 3: cells (geometry on CPU, text batched) ----------------------
    t0 = time.perf_counter()
    cell_jobs, cell_targets, n_tables, n_cells, n_derived = [], [], 0, 0, 0
    if detect_table_cells:
        for r in all_regions:
            if r.label != "table" or r.skipped:
                continue
            n_tables += 1
            crop = images[r.page_index].crop(tuple(r.bbox_px))
            find_cells(r, crop, images[r.page_index].size,
                       borderless_fallback=borderless_fallback)
            n_cells += len(r.cells)

            if r.cells and cell_text == "derive":
                n_derived += derive_cell_text(r)
            elif r.cells and cell_text == "batch":
                for c in r.cells:
                    cell_crop = crop.crop(tuple(c["bbox_px"]))
                    cell_crop.load()   # PIL crop is lazy — materialize before
                                       # the parent is closed
                    cell_jobs.append((cell_crop, "Text Recognition:",
                                      tok_budget("cell")))
                    cell_targets.append(c)
            crop.close()

        mode = {"derive": "text derived from table HTML (no model calls)",
                "batch": "per-cell OCR", "none": "boxes only"}[cell_text]
        print(f"[3/3] cells     : {n_cells} cells in {n_tables} table(s) — {mode}")
        if cell_jobs:
            for cell, text in zip(cell_targets, glm.run(cell_jobs, "  cells")):
                cell["text"] = text
                cell["text_source"] = "ocr"
            for job in cell_jobs:
                job[0].close()
        if n_derived:
            print(f"  derived {n_derived}/{n_cells} cell texts in "
                  f"{time.perf_counter() - t0:.2f}s")
        for r in all_regions:
            if r.cells:
                r.cell_html = cells_to_html(r.cells)
    cell_s = time.perf_counter() - t0
    if timings and detect_table_cells:
        timings.record("stage 3 cells", cell_s,
                       f"{n_cells} cells, mode={cell_text}")

    for img in images:
        img.close()
    all_regions.sort(key=lambda r: (r.page_index, r.seq_index))

    total = time.perf_counter() - t_all
    print(f"\nTOTAL {total:.1f}s for {len(page_paths)} page(s) = "
          f"{total / max(1, len(page_paths)):.1f}s/page"
          f"   [regions {ocr_s:.1f}s | cells {cell_s:.1f}s]")
    return all_regions


def run_fullpage(page_paths: list[Path], glm: GlmOCR) -> list[dict]:
    """Whole-page OCR with no layout stage — a baseline for comparison."""
    images = [Image.open(p).convert("RGB") for p in page_paths]
    texts = glm.run([(im, "Text Recognition:", MAX_TOK_FULLPAGE) for im in images],
                    "  fullpage")
    for img in images:
        img.close()
    return [{"page_index": i, "markdown": t} for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(regions: list[Region]) -> str:
    lines = ["=" * 104,
             "REGIONS — bbox from PP-DocLayoutV3, text from GLM-OCR, "
             "cells from morphology",
             "=" * 104,
             f"{'pg':>2} {'#':>3} {'label':<11} {'det':>4} {'cells':>5}  "
             f"{'bbox norm  l,t,r,b':<32} text",
             "-" * 104]
    for r in regions:
        bbox = ",".join(f"{v:.3f}" for v in r.bbox_norm)
        preview = f"[{r.skipped}]" if r.skipped else r.text.replace("\n", " / ")
        if len(preview) > 44:
            preview = preview[:41] + "..."
        n_cells = str(len(r.cells)) if r.cells else ("-" if r.label != "table" else "0")
        lines.append(f"{r.page_index:>2} {r.seq_index:>3} {r.label:<11} "
                     f"{r.layout_confidence:>4.2f} {n_cells:>5}  {bbox:<32} {preview}")

    with_text = [r for r in regions if r.text]
    lines += ["-" * 104,
              f"regions={len(regions)}  with_text={len(with_text)}  "
              f"empty={len(regions) - len(with_text)}  "
              f"chars={sum(len(r.text) for r in regions)}  "
              f"table_cells={sum(len(r.cells) for r in regions)}"]

    for r in regions:
        if r.label != "table":
            continue
        lines += ["", "=" * 70,
                  f"TABLE p{r.page_index} #{r.seq_index}  bbox_px={r.bbox_px}",
                  "-" * 70, "[region-level GLM-OCR output]", r.text or "[empty]"]
        if r.cells:
            method = r.cells[0].get("method", "?")
            lines.append(f"\n[cell-level: {len(r.cells)} cells via {method}]")
            for c in r.cells:
                lines.append(
                    f"  r{c['row']:<2} c{c['col']:<2} "
                    f"span({c['rowspan']}x{c['colspan']}) px={c['bbox_px']} "
                    f"norm={c['bbox_norm_page']} -> {(c.get('text') or '')!r}")
            lines += ["", "[reassembled HTML]", r.cell_html]
    return "\n".join(lines)


def draw_overlays(page_paths: list[Path], regions: list[Region],
                  out_dir: Path) -> list[Path]:
    """Thick coloured box per region; thin red box per detected table cell."""
    by_page: dict[int, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page_index, []).append(r)

    dests = []
    for i, path in enumerate(page_paths):
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for r in by_page.get(i, []):
            colour = (190, 190, 190) if r.skipped else REGION_COLORS.get(
                r.label, (60, 60, 60))
            draw.rectangle(r.bbox_px, outline=colour, width=3)
            tag = f"#{r.seq_index} {r.label} {r.layout_confidence:.2f}"
            y = max(0, r.bbox_px[1] - 14)
            draw.rectangle([r.bbox_px[0], y, r.bbox_px[0] + 7 * len(tag), y + 14],
                           fill=colour)
            draw.text((r.bbox_px[0] + 2, y + 2), tag, fill=(255, 255, 255))

            ox, oy = r.bbox_px[0], r.bbox_px[1]
            for cell in r.cells:
                x1, y1, x2, y2 = cell["bbox_px"]
                draw.rectangle([ox + x1, oy + y1, ox + x2, oy + y2],
                               outline=(255, 0, 0), width=1)

        dest = out_dir / f"{path.stem}_overlay.png"
        img.save(dest)
        img.close()
        dests.append(dest)
    return dests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GLM-OCR invoice extractor — region boxes, text, and table cells.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Speed, in order of impact: --cell-text derive (removes every "
               "per-cell model call), then --dpi 150, then --batch-size.",
    )
    p.add_argument("input", nargs="?", default=PDF_PATH,
                   help=f"Path to the invoice PDF. Defaults to PDF_PATH at the "
                        f"top of this file (currently: {PDF_PATH})")
    p.add_argument("--image", action="store_true",
                   help="Treat input as a single image instead of a PDF")
    p.add_argument("--out", default=OUT_DIR, help="Output directory")
    p.add_argument("--dpi", type=int, default=DPI,
                   help="Rasterization DPI (150 = faster, 300 = small fonts)")
    p.add_argument("--pages",
                   default=",".join(str(x) for x in PAGES) if PAGES else None,
                   help="Comma-separated 0-based page indices, e.g. 0,1")

    g = p.add_argument_group("speed")
    g.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Crops per forward pass (12-16 if VRAM allows)")
    g.add_argument("--max-crop-px", type=int, default=1600,
                   help="Downscale crops whose longest side exceeds this")
    g.add_argument("--cell-text", choices=["derive", "batch", "none"],
                   default=CELL_TEXT,
                   help="derive = map text from the table-level HTML, no extra "
                        "model calls (fastest, recommended); batch = OCR each "
                        "cell (most accurate, slowest); none = boxes only")
    g.add_argument("--skip-labels", default="figure",
                   help="Comma-separated region labels to skip ('' for none)")
    g.add_argument("--fullpage", action="store_true",
                   help="Also run the whole-page OCR baseline (slower)")

    g2 = p.add_argument_group("detection")
    g2.add_argument("--layout-threshold", type=float, default=0.3,
                    help="Lower detects more regions")
    g2.add_argument("--no-cells", action="store_true",
                    help="Skip stage 3 entirely")
    g2.add_argument("--no-borderless-fallback", action="store_true",
                    help="Do not try whitespace projection on borderless tables")
    g2.add_argument("--min-region-px", type=int, default=20,
                    help="Skip regions smaller than this in either dimension")
    g2.add_argument("--device", default="auto", help="auto | cuda | cpu")
    g2.add_argument("--no-overlay", action="store_true",
                    help="Skip annotated PNG output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: input not found: {src}\n", file=sys.stderr)
        print("Fix it either way:", file=sys.stderr)
        print("  1. edit PDF_PATH at the top of this file, or", file=sys.stderr)
        print(f'  2. pass a path:  python {Path(__file__).name} '
              f'"C:\\path\\to\\invoice.pdf"', file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = Timings()

    # --- rasterize on a worker thread while the models load on the main one.
    # Rendering is pure CPU and model loading is mostly disk + PCIe transfer,
    # so the two overlap almost perfectly — the render becomes close to free.
    def _rasterize() -> list[Path]:
        if args.image:
            dest = out_dir / "page_000.png"
            Image.open(src).convert("RGB").save(dest)
            print(f"using image {src.name}")
            return [dest]
        pages = [int(x) for x in args.pages.split(",")] if args.pages else None
        return render_pdf(src, out_dir, args.dpi, pages)

    with ThreadPoolExecutor(max_workers=1) as pool:
        t0 = time.perf_counter()
        render_future = pool.submit(_rasterize)

        detector = LayoutDetector(device=args.device,
                                  threshold=args.layout_threshold)
        detector.start()
        glm = GlmOCR(batch_size=args.batch_size, max_crop_px=args.max_crop_px)
        glm.start()
        load_s = time.perf_counter() - t0

        page_paths = render_future.result()

    timings.record("load models", load_s, "(one-time; overlapped with render)")
    if not page_paths:
        print("ERROR: nothing to process", file=sys.stderr)
        return 3

    skip_labels = {s.strip() for s in args.skip_labels.split(",") if s.strip()}

    # --- run ----------------------------------------------------------------
    t0 = time.perf_counter()
    regions = run_pipeline(
        page_paths, detector, glm,
        min_px=args.min_region_px,
        skip_labels=skip_labels,
        detect_table_cells=not args.no_cells,
        cell_text=args.cell_text,
        borderless_fallback=not args.no_borderless_fallback,
        timings=timings,
    )
    timings.record("stages 1-3", time.perf_counter() - t0,
                   f"{len(regions)} regions")

    report = format_report(regions)
    print("\n" + report)

    t0 = time.perf_counter()
    (out_dir / "regions.json").write_text(
        json.dumps([asdict(r) for r in regions], indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "summary.txt").write_text(report, encoding="utf-8")

    if args.fullpage:
        print("\nfull-page baseline:")
        (out_dir / "fullpage.json").write_text(
            json.dumps(run_fullpage(page_paths, glm), indent=2, ensure_ascii=False),
            encoding="utf-8")

    if not args.no_overlay:
        for dest in draw_overlays(page_paths, regions, out_dir):
            print(f"overlay: {dest}")
    timings.record("write output", time.perf_counter() - t0)

    print("\n" + timings.report(len(page_paths)))
    print(f"\nArtifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
