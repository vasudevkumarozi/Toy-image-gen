# Pipeline checking UI

A one-page Streamlit app for running the full pipeline (fetch → classify →
generate → build_wide_summary) from a browser instead of four separate
terminal commands — built so you can confirm everything works, watching
live progress and logs, before ever running it unattended on a deploy
instance.

## Run it

```bash
pip install -r requirements.txt   # adds streamlit + python-dotenv
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. On an instance, run the same command
and either port-forward (`ssh -L 8501:localhost:8501 <instance>`) or open
the port in the firewall — see [`../DEPLOY.md`](../DEPLOY.md) for the
actual instance commands (start/stop/tunnel) already set up for this
project's GCP instance.

## What it does

1. **Sidebar** — checks `OZI_API_KEY`, `GCP_PROJECT_ID`,
   `GOOGLE_APPLICATION_CREDENTIALS`, and `GCS_BUCKET` are set (from your
   `.env`) before letting you run anything; lets you set `--workers` and
   `--overwrite`.
2. **Upload** a product list (`.xlsx`/`.csv` with a `Product ID` or
   `Admin Panel Link` column) — same file format the CLI scripts take.
3. **Run pipeline** — each of the 4 steps runs as its own subprocess, with
   live-streamed logs and a progress bar (backed by the same
   `<out>.status.json` files the pipeline already writes for unattended
   VM monitoring). If a step fails, the run stops there — nothing
   downstream executes — and the failing step's full log stays visible.
4. **Result** — a download button for `products_6_images.xlsx`, plus a
   quick breakdown of how many slots came back `Existing` / `Generated` /
   `Generated (needs review)`.

Each run gets its own folder under `ui_runs/<timestamp>/` (gitignored) —
re-running the same input resumes from checkpoint rather than redoing
completed work or re-paying for API calls already made.
