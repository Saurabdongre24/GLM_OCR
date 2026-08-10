# GLM-OCR Invoice Demo — Colab

Self-contained. Three files, one command, one link.

```
colab_app/
├── server.py            <- run this
├── glm_ocr_invoice.py   <- the pipeline (all the real logic)
├── static/index.html    <- the web page
└── requirements.txt
```

## Run it

Upload this whole folder to Colab (Runtime → Change runtime type → **T4 GPU**),
then in the **terminal**:

```bash
cd colab_app
pip install -r requirements.txt
python server.py
```

It prints a link. Open it, drop your invoice PDF on the page.

> If `pip install` ends with `ImportError: cannot import name '_Ink'`, run
> `pip install --force-reinstall --no-cache-dir pillow` and open a fresh
> terminal. Colab imported the old pillow before the upgrade replaced it.

## The ngrok token

Open `server.py`. It is the first setting in the file:

```python
NGROK_TOKEN = ""        # <- paste your token between the quotes
```

Get one free (no card) at
<https://dashboard.ngrok.com/get-started/your-authtoken>.

Or set it without editing the file:

```bash
export NGROK_TOKEN=xxxxxxxxxxxx
python server.py
```

**You can also leave it empty.** The server then downloads `cloudflared` and
uses that instead — no account, no token. Try that first; only switch to ngrok
if cloudflared is blocked on your network.

### `ERR_NGROK_105` — "does not look like a proper ngrok authtoken"

You copied the wrong credential. An ngrok **authtoken** is roughly 49
characters with **no prefix**. Anything shaped like `cr_…`, `sk_…` or
`ak_…` is an API key or a token from a different service.

Get the right one from the **Your Authtoken** page:
<https://dashboard.ngrok.com/get-started/your-authtoken> — not the API-keys
page.

Or sidestep it: set `NGROK_TOKEN = ""` and use cloudflared. The server now
detects a malformed token before contacting ngrok, and falls back to
cloudflared automatically if ngrok fails for any reason, so a bad token no
longer costs you the link.

To force one or the other, set `TUNNEL` just below the token:
`"auto"` (default) · `"ngrok"` · `"cloudflared"` · `"none"`.

## Using it

1. Wait for the status pill (top right) to turn green — `models ready`.
   First run downloads ~2.2 GB and takes about 40 seconds.
2. Drop an invoice PDF on the page. **The upload happens in the browser** —
   nothing to do in the terminal.
3. Watch the three stages light up, then explore the result:
   - click any box on the page to see its text and coordinates
   - toggle **Stage 1 · regions** / **Stage 3 · cells** to see what each adds
   - **Tables** — extracted tables with merged cells preserved
   - **Stages** — what each stage produced on this document
   - **Speed** — per-stage timing

Settings on the upload screen: **DPI** (150 fast / 175 balanced / 300 small
fonts), **Cell text** (`derive` fast, `batch` accurate), **Batch size** (drop to
6 if you hit CUDA out-of-memory), **Pages** (e.g. `0,1`).

## Anyone with the link can use it

Both tunnels are public URLs with no password. While the server runs, anyone
holding the link can upload documents to your GPU session. Stop it with
`Ctrl+C` when you are done demoing.

## Troubleshooting

| Problem | Fix |
|---|---|
| Status pill red | The page shows the reason; usually a `transformers` version issue |
| `CUDA out of memory` | Set **Batch size** to 6 on the upload screen |
| No link printed | Add `NGROK_TOKEN`, or check `cloudflared.log` in this folder |
| Page loads, buttons dead | You are on an old `index.html` — the current one uses relative URLs so it works behind a tunnel |
| Slow | Check the **Speed** tab. Lower DPI to 150; keep Cell text on `derive` |

## Note on this folder

`glm_ocr_invoice.py` here is a **copy** of `scripts/glm_ocr_invoice.py`, so the
folder can be uploaded on its own. If you change the pipeline in `scripts/`,
copy it over again — otherwise the two will drift apart.
