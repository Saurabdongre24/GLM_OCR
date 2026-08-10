#!/usr/bin/env python
"""GLM-OCR invoice demo — single-command web app for Colab.

Run it from the Colab terminal:

    pip install -r requirements.txt
    python server.py

It prints a public link. Open it, drop an invoice PDF on the page, and the
result comes back with a box around every region and every table cell.

    colab_app/
      server.py            <- run this
      glm_ocr_invoice.py   <- the pipeline (all the real logic lives here)
      static/index.html    <- the web page
      requirements.txt

Put your ngrok token in NGROK_TOKEN below. If you leave it empty the server
falls back to cloudflared, which needs no account at all.
"""

from __future__ import annotations

# ===========================================================================
#  >>>>>  PUT YOUR NGROK TOKEN HERE  <<<<<
#
#  Get one free (no card) at:
#      https://dashboard.ngrok.com/get-started/your-authtoken
#
#  Leave it as "" to use cloudflared instead — no account, no token needed.
#  You can also set it without editing this file:
#      export NGROK_TOKEN=xxxxxxxx      (then: python server.py)
# ===========================================================================

NGROK_TOKEN = ""

TUNNEL = "auto"   # "auto" = ngrok if a token exists, else cloudflared
                  # "ngrok" | "cloudflared" | "none" (local only)
PORT = 8000

# ===========================================================================

import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Windows/!UTF-8 consoles cannot encode the dashes the pipeline prints, and one
# such line would kill the server. Harmless on Linux; keep it for portability.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent

# The pipeline sits next to this file. Nothing below reimplements any of it —
# this server only calls into it.
sys.path.insert(0, str(HERE))
import glm_ocr_invoice as pipeline  # noqa: E402

WORK = HERE / "_runs"
WORK.mkdir(exist_ok=True)
STATIC = HERE / "static"


# ---------------------------------------------------------------------------
# Public link
# ---------------------------------------------------------------------------

def _looks_like_ngrok_token(token: str) -> bool:
    """ngrok authtokens are a long unprefixed alphanumeric string.

    Anything with a `xx_` prefix is a different credential — an API key, or a
    token from another service entirely. Catching this here turns a wall of
    ngrok stack traces into one actionable line.
    """
    return len(token) >= 30 and not re.match(r"^[a-z]{2,6}_", token)


def _start_ngrok(port: int, token: str) -> str | None:
    if not token:
        print("! no NGROK_TOKEN set — put one at the top of server.py")
        return None
    if not _looks_like_ngrok_token(token):
        print(f"! that does not look like an ngrok authtoken: {token[:12]}…")
        print("  ngrok authtokens are ~49 characters with no 'xx_' prefix.")
        print("  Copy it from https://dashboard.ngrok.com/get-started/your-authtoken")
        print("  (the AUTHTOKEN page — not API keys, not another service)")
        return None
    try:
        from pyngrok import ngrok
    except ImportError:
        print("! pyngrok not installed —  pip install pyngrok")
        return None
    try:
        ngrok.set_auth_token(token)
        ngrok.kill()                         # drop tunnels from earlier runs
        return ngrok.connect(port).public_url
    except Exception as exc:                 # noqa: BLE001
        print(f"! ngrok failed: {str(exc)[:200]}")
        return None


def _start_cloudflared(port: int) -> str | None:
    exe = HERE / "cloudflared"
    if not exe.exists():
        print("· downloading cloudflared (one-off, no account needed)…")
        url = ("https://github.com/cloudflare/cloudflared/releases/latest/"
               "download/cloudflared-linux-amd64")
        try:
            subprocess.run(["wget", "-q", "-O", str(exe), url], check=True)
            exe.chmod(0o755)
        except Exception as exc:             # noqa: BLE001
            print(f"! could not download cloudflared: {exc}")
            return None

    log = HERE / "cloudflared.log"
    print("· starting cloudflared tunnel…")
    subprocess.Popen(
        [str(exe), "tunnel", "--url", f"http://localhost:{port}",
         "--no-autoupdate"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )
    for _ in range(40):                      # it prints the URL when ready
        time.sleep(1)
        try:
            m = re.search(r"https://[-\w]+\.trycloudflare\.com",
                          log.read_text(errors="ignore"))
            if m:
                return m.group(0)
        except FileNotFoundError:
            pass
    print("! cloudflared did not report a URL — see cloudflared.log")
    return None


def start_tunnel(port: int) -> str | None:
    """Expose `port` publicly. Returns the URL, or None if it stayed local.

    A Colab *terminal* cannot use the notebook's JS port proxy, so a tunnel is
    the only way to reach the server from your browser. If ngrok is asked for
    but fails — bad token, quota, network — we fall through to cloudflared
    rather than leaving you with no link at all.
    """
    token = (os.environ.get("NGROK_TOKEN") or NGROK_TOKEN).strip()

    if TUNNEL == "none":
        return None
    if TUNNEL == "cloudflared":
        return _start_cloudflared(port)
    if TUNNEL == "ngrok":
        return _start_ngrok(port, token)

    # "auto": prefer ngrok only if a token was supplied, but never let an
    # ngrok problem cost you the link — cloudflared needs no account.
    if token:
        url = _start_ngrok(port, token)
        if url:
            return url
        print("· falling back to cloudflared (no account needed)")
    return _start_cloudflared(port)


# ---------------------------------------------------------------------------
# Models — loaded once, shared by every upload
# ---------------------------------------------------------------------------

class Models:
    def __init__(self) -> None:
        self.detector = None
        self.glm = None
        self.state = "cold"          # cold | loading | ready | failed
        self.error = ""
        self.load_seconds = 0.0
        self.lock = threading.Lock()  # one job on the GPU at a time

    def load(self, batch_size: int = 12) -> None:
        if self.state in ("loading", "ready"):
            return
        self.state = "loading"
        t0 = time.perf_counter()
        try:
            print("· loading models (first run downloads ~2.2 GB)…")
            self.detector = pipeline.LayoutDetector(device="auto", threshold=0.3)
            self.detector.start()
            self.glm = pipeline.GlmOCR(batch_size=batch_size)
            self.glm.start()
            self.load_seconds = round(time.perf_counter() - t0, 1)
            self.state = "ready"
            print(f"· models ready in {self.load_seconds}s — upload a PDF")
        except Exception as exc:             # noqa: BLE001
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"! model load FAILED — {self.error}")
            traceback.print_exc()


MODELS = Models()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Background, so the page is reachable immediately and can show progress
    # instead of hanging on a blank screen for 40 seconds.
    threading.Thread(target=MODELS.load, daemon=True).start()
    yield


app = FastAPI(title="GLM-OCR Invoice Demo", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

JOBS: dict[str, dict] = {}


class LogTee(io.TextIOBase):
    """Tees the pipeline's stdout into a job log so the page can show progress.

    The pipeline reports by printing. Teeing keeps it a clean CLI tool instead
    of threading a callback through it for the demo's benefit.
    """

    def __init__(self, job: dict) -> None:
        self.job = job
        self.buf = ""

    def write(self, s: str) -> int:
        sys.__stdout__.write(s)
        self.buf += s
        while "\n" in self.buf or "\r" in self.buf:
            idx = min(i for i in (self.buf.find("\n"), self.buf.find("\r")) if i >= 0)
            line, self.buf = self.buf[:idx].rstrip(), self.buf[idx + 1:]
            if line:
                self.job["log"].append(line)
                del self.job["log"][:-400]
        return len(s)

    def flush(self) -> None:
        sys.__stdout__.flush()


def run_job(job_id: str, pdf_path: Path, opts: dict) -> None:
    job = JOBS[job_id]
    out_dir = pdf_path.parent
    try:
        if MODELS.state != "ready":
            job["log"].append("waiting for models to finish loading…")
        while MODELS.state == "loading":
            time.sleep(0.5)
        if MODELS.state != "ready":
            raise RuntimeError(f"models unavailable: {MODELS.error or MODELS.state}")

        with redirect_stdout(LogTee(job)):
            job["phase"] = "rendering"
            page_paths = pipeline.render_pdf(
                pdf_path, out_dir, opts["dpi"], opts["pages"] or None)
            job["pages"] = len(page_paths)

            job["phase"] = "running"
            timings = pipeline.Timings()
            # Serialise GPU access — concurrent generate() calls would contend
            # and could OOM if two people upload at once.
            with MODELS.lock:
                MODELS.glm.batch_size = opts["batch_size"]
                regions = pipeline.run_pipeline(
                    page_paths, MODELS.detector, MODELS.glm,
                    skip_labels={s for s in opts["skip_labels"] if s},
                    cell_text=opts["cell_text"], timings=timings)

            job["phase"] = "drawing"
            pipeline.draw_overlays(page_paths, regions, out_dir)

        job["result"] = build_payload(regions, page_paths, timings, opts)
        job["phase"] = "done"
    except Exception as exc:                 # noqa: BLE001
        job["phase"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["log"].append(f"ERROR — {job['error']}")
        traceback.print_exc()


def build_payload(regions, page_paths, timings, opts) -> dict:
    """Shape the pipeline output for the browser.

    Boxes stay in normalized 0–1 coordinates so the page can position them as
    percentages without knowing any pixel dimensions.
    """
    by_page: dict[int, list] = {}
    for r in regions:
        d = asdict(r)
        d["has_text"] = bool(r.text)
        by_page.setdefault(r.page_index, []).append(d)

    tables = [r for r in regions if r.label == "table" and r.cells]
    return {
        "pages": [{"index": i, "regions": by_page.get(i, [])}
                  for i in range(len(page_paths))],
        "summary": {
            "pages": len(page_paths),
            "regions": len(regions),
            "with_text": sum(1 for r in regions if r.text),
            "tables": sum(1 for r in regions if r.label == "table"),
            "cells": sum(len(r.cells) for r in regions),
            "merged_cells": sum(1 for r in regions for c in r.cells
                                if c.get("colspan", 1) > 1 or c.get("rowspan", 1) > 1),
            "characters": sum(len(r.text) for r in regions),
            "cell_method": (tables[0].cells[0].get("method") if tables else None),
        },
        "timings": {
            "phases": [{"name": k, "seconds": round(v, 2),
                        "note": timings.notes.get(k, "")}
                       for k, v in timings.phases.items()],
            "total": round(timings.elapsed(), 2),
            "per_page": round(timings.elapsed() / max(1, len(page_paths)), 2),
            "model_load": MODELS.load_seconds,
        },
        "options": opts,
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def build_kv(payload: dict) -> dict:
    """Key/value view of everything that was read.

    Where the keys come from, in order of reliability:

    1. **Table rows.** In an invoice the left-hand column is nearly always the
       label and the rest of the row is its value — `Port Code | INMDD6`,
       `GSTIN/TYPE | 23AAACI3924J1ZS/G`. Merged value cells collapse into one
       string, which is exactly what you want.
    2. **`key_value` regions**, split on the first colon.

    Duplicate labels get suffixed (`Total`, `Total (2)`) rather than silently
    overwriting each other — an invoice can legitimately repeat a label.
    """
    fields: dict[str, str] = {}
    tables: list[dict] = []

    def put(key: str, value: str) -> None:
        key, value = _clean(key), _clean(value)
        if not key or not value or len(key) > 60:
            return
        if key in fields:
            if fields[key] == value:
                return                       # same pair twice — ignore
            n = 2
            while f"{key} ({n})" in fields:
                n += 1
            key = f"{key} ({n})"
        fields[key] = value

    for page in payload["pages"]:
        for r in page["regions"]:
            if r["label"] == "table" and r.get("cells"):
                by_row: dict[int, list] = {}
                for c in r["cells"]:
                    by_row.setdefault(c["row"], []).append(c)

                rows = []
                for key in sorted(by_row):
                    cells = sorted(by_row[key], key=lambda c: c["col"])
                    texts = [_clean(c.get("text", "")) for c in cells]
                    rows.append(texts)
                    if len(texts) >= 2:
                        put(texts[0], " ".join(t for t in texts[1:] if t))

                tables.append({
                    "page": r["page_index"] + 1,
                    "region": r["seq_index"],
                    "bbox": r["bbox_norm"],
                    "rows": rows,
                })

            elif r["label"] == "key_value" and r.get("text"):
                for line in r["text"].splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        put(k, v)

    return {
        "document": payload["options"].get("filename"),
        "pages": payload["summary"]["pages"],
        "fields": fields,
        "field_count": len(fields),
        "tables": tables,
        "text_by_region": [
            {"page": r["page_index"] + 1, "index": r["seq_index"],
             "label": r["label"], "text": r["text"]}
            for p in payload["pages"] for r in p["regions"] if r.get("text")
        ],
    }


def build_boxes_md(payload: dict) -> str:
    """Every bounding box, as a readable Markdown document.

    Coordinates are the normalized 0-1 page space — the form worth storing,
    since it survives a change of DPI.
    """
    s = payload["summary"]
    out = [
        f"# Bounding boxes — {payload['options'].get('filename') or 'document'}",
        "",
        f"- Pages: **{s['pages']}**",
        f"- Regions: **{s['regions']}** ({s['with_text']} with text)",
        f"- Tables: **{s['tables']}** · cells: **{s['cells']}** "
        f"({s['merged_cells']} merged)",
        f"- Characters read: **{s['characters']:,}**",
        "",
        "Coordinates are `left, top, right, bottom`, normalized 0–1 with the "
        "origin at the top-left of the page.",
        "",
    ]

    for page in payload["pages"]:
        out += [f"## Page {page['index'] + 1}", "",
                "| # | label | conf | left | top | right | bottom | text |",
                "|---|-------|-----:|-----:|----:|------:|-------:|------|"]
        for r in page["regions"]:
            l, t, rt, b = r["bbox_norm"]
            txt = _clean(r["text"])[:70] or (f"_{r['skipped']}_" if r["skipped"] else "")
            out.append(
                f"| {r['seq_index']} | `{r['label']}` | {r['layout_confidence']:.2f} "
                f"| {l:.4f} | {t:.4f} | {rt:.4f} | {b:.4f} "
                f"| {txt.replace('|', '\\|')} |")
        out.append("")

        for r in page["regions"]:
            if not r.get("cells"):
                continue
            out += [f"### Table cells — region #{r['seq_index']} "
                    f"({len(r['cells'])} cells, via {r['cells'][0].get('method','?')})",
                    "",
                    "| row | col | span | left | top | right | bottom | text |",
                    "|----:|----:|------|-----:|----:|------:|-------:|------|"]
            for c in r["cells"]:
                l, t, rt, b = c["bbox_norm_page"]
                span = (f"{c['rowspan']}×{c['colspan']}"
                        if c["rowspan"] > 1 or c["colspan"] > 1 else "—")
                txt = _clean(c.get("text", ""))[:50]
                out.append(
                    f"| {c['row']} | {c['col']} | {span} "
                    f"| {l:.4f} | {t:.4f} | {rt:.4f} | {b:.4f} "
                    f"| {txt.replace('|', '\\|')} |")
            out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"models": MODELS.state, "error": MODELS.error,
            "load_seconds": MODELS.load_seconds}


@app.post("/api/jobs")
async def create_job(file: UploadFile, dpi: int = 175, batch_size: int = 12,
                     cell_text: str = "derive", pages: str = "",
                     skip_labels: str = "figure") -> JSONResponse:
    name = (file.filename or "upload").lower()
    if not name.endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Upload a PDF or an image (.pdf, .png, .jpg)")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK / job_id
    job_dir.mkdir(parents=True)
    dest = job_dir / (file.filename or "input.pdf")
    dest.write_bytes(await file.read())

    if not name.endswith(".pdf"):            # image input -> single page
        from PIL import Image
        Image.open(dest).convert("RGB").save(job_dir / "page_000.png")

    opts = {
        "dpi": max(72, min(400, dpi)),
        "batch_size": max(1, min(32, batch_size)),
        "cell_text": cell_text if cell_text in ("derive", "batch", "none") else "derive",
        "pages": [int(p) for p in pages.split(",") if p.strip().isdigit()],
        "skip_labels": [s.strip() for s in skip_labels.split(",")],
        "filename": file.filename,
    }
    JOBS[job_id] = {"id": job_id, "phase": "queued", "log": [], "pages": 0,
                    "result": None, "error": "", "dir": str(job_dir),
                    "started": time.time()}
    threading.Thread(target=run_job, args=(job_id, dest, opts), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return {"id": job_id, "phase": job["phase"], "error": job["error"],
            "pages": job["pages"],
            "elapsed": round(time.time() - job["started"], 1),
            "log": job["log"][since:], "log_total": len(job["log"]),
            "result": job["result"] if job["phase"] == "done" else None}


@app.get("/api/jobs/{job_id}/page/{page}")
def page_image(job_id: str, page: int, overlay: bool = False):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    stem = f"page_{page:03d}"
    path = Path(job["dir"]) / (f"{stem}_overlay.png" if overlay else f"{stem}.png")
    if not path.exists():
        raise HTTPException(404, f"No such page: {path.name}")
    return FileResponse(path, media_type="image/png")


def _finished(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    if not job.get("result"):
        raise HTTPException(409, f"Job is not finished yet (phase: {job['phase']})")
    return job["result"]


def _stem(payload: dict) -> str:
    name = payload["options"].get("filename") or "document"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem)[:60] or "document"


@app.get("/api/jobs/{job_id}/export/kv.json")
def export_kv(job_id: str):
    payload = _finished(job_id)
    return JSONResponse(
        build_kv(payload),
        headers={"Content-Disposition":
                 f'attachment; filename="{_stem(payload)}_extracted.json"'},
    )


@app.get("/api/jobs/{job_id}/export/boxes.md")
def export_boxes(job_id: str):
    from fastapi.responses import PlainTextResponse

    payload = _finished(job_id)
    return PlainTextResponse(
        build_boxes_md(payload),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="{_stem(payload)}_boxes.md"'},
    )


@app.get("/api/jobs/{job_id}/export/raw.json")
def export_raw(job_id: str):
    """Everything the pipeline produced, unshaped — for downstream code."""
    payload = _finished(job_id)
    return JSONResponse(
        payload,
        headers={"Content-Disposition":
                 f'attachment; filename="{_stem(payload)}_full.json"'},
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = JOBS.pop(job_id, None)
    if job:
        shutil.rmtree(job["dir"], ignore_errors=True)
    return {"deleted": bool(job)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    public = start_tunnel(PORT)

    print("\n" + "=" * 68)
    if public:
        print(f"  OPEN THIS LINK:   {public}")
        print("  (anyone with it can reach this server — close the terminal "
              "when done)")
    else:
        print(f"  Running locally:  http://localhost:{PORT}")
        print("  No public link. On Colab, add your NGROK_TOKEN at the top of")
        print("  server.py, or let it fall back to cloudflared.")
    print("=" * 68)
    print("  Models load in the background — the page shows when they're ready.")
    print("  Upload your PDF on the page itself.\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
