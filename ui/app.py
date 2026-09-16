"""
OZi Toys Image Pipeline — checking UI
=======================================================================
A Streamlit front-end for the 4-step pipeline (fetch -> classify ->
generate -> build_wide_summary), built specifically so you can upload a
product list and WATCH the run happen — live logs, a progress bar per
step (backed by the same status.json files the pipeline already writes
for unattended VM runs), and a clear red stop on the first failed step
instead of silently continuing.

The actual pipeline run is a fully DETACHED background process
(scripts/run_pipeline_all.py, launched with start_new_session=True) —
NOT something that runs inside this Streamlit script's own execution.
A real run was silently abandoned after step 2 when the browser tab's
SSH tunnel dropped: the step-2 subprocess had already finished cleanly,
but the UI's own script execution (which was sequencing steps 1-4
in-line) got torn down before it reached step 3, and step 3 never
started. Because the run is now a separate OS process whose parent is
init, not Streamlit, it survives regardless of what happens to any
browser tab, tunnel, or session — and every run's progress lives on
disk (pipeline_status.json + the same per-step *.status.json files),
so reopening the page later shows the real current state instead of a
blank "nothing has ever run" screen.

This is meant to be run BOTH locally (to sanity-check the pipeline before
ever touching the deploy instance) and later on the instance itself, so
"does this actually work" has a UI answer, not just a terminal you have
to SSH into and tail.

Run it with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RUNS_DIR = REPO_ROOT / "ui_runs"
RUNS_DIR.mkdir(exist_ok=True)

load_dotenv(REPO_ROOT / ".env")

st.set_page_config(page_title="OZi Toys Image Pipeline", layout="wide")

STAGE_LABELS = {
    "fetch": ("Step 1/4 — Fetching product details", "products_detail.csv"),
    "classify": ("Step 2/4 — Classifying existing images", "classification_result.csv"),
    "generate": ("Step 3/4 — Generating missing images", "final_output.xlsx"),
    "summary": ("Step 4/4 — Building the final summary sheet", None),
    "done": ("Pipeline complete", None),
    "failed": ("Pipeline failed", None),
}
STAGE_ORDER = ["fetch", "classify", "generate", "summary", "done"]


def env_check() -> dict:
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return {
        "OZI_API_KEY": bool(os.environ.get("OZI_API_KEY")),
        "GCP_PROJECT_ID": bool(os.environ.get("GCP_PROJECT_ID")),
        "GOOGLE_APPLICATION_CREDENTIALS (file exists)": bool(creds_path) and os.path.exists(creds_path),
        "GCS_BUCKET (optional — enables auto-upload)": bool(os.environ.get("GCS_BUCKET")),
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def list_runs() -> list:
    """Every run under ui_runs/ that has a pipeline_status.json, newest
    first — this is what makes a run reattachable after a fresh page
    load, since it's the only source of truth (not session_state)."""
    runs = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        status = read_json(d / "pipeline_status.json")
        if not status:
            continue
        runs.append((d.name, status))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs


def launch_run(input_path: Path, run_dir: Path, workers: int, overwrite: bool) -> None:
    """Starts run_pipeline_all.py fully detached — start_new_session=True
    puts it in its own process group/session so it does NOT receive a
    SIGHUP if this Streamlit process (or its controlling terminal/SSH
    session) goes away, the same guarantee `nohup` gives a shell command."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "run_pipeline_all.py"),
          "--run-dir", str(run_dir), "--input", str(input_path),
          "--workers", str(workers)]
    if overwrite:
        cmd.append("--overwrite")
    log_path = run_dir / "pipeline.log"
    with open(log_path, "a") as log_f:
        subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=os.environ.copy(), start_new_session=True,
        )


def render_step_progress(run_dir: Path, status_filename: str, label: str) -> None:
    data = read_json(run_dir / f"{status_filename}.status.json")
    total = data.get("total", 0)
    done = data.get("done", 0)
    if total:
        counts = {k: v for k, v in data.items()
                 if k not in ("total", "done", "started_at", "updated_at", "finished_at")}
        counts_str = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        st.progress(min(done / total, 1.0),
                   text=f"{label}: {done}/{total}" + (f" — {counts_str}" if counts_str else ""))
    else:
        st.caption(f"{label}: waiting for progress data...")


def render_run(run_id: str) -> None:
    run_dir = RUNS_DIR / run_id
    status = read_json(run_dir / "pipeline_status.json")
    stage = status.get("stage", "unknown")
    label, status_filename = STAGE_LABELS.get(stage, (f"Unknown stage: {stage}", None))

    if stage == "done":
        st.success(f"✅ {label}")
    elif stage == "failed":
        st.error(f"❌ {label}: {status.get('error', 'unknown error')}")
    else:
        st.info(f"⏳ {label}...")

    # Progress bars for every step up to and including the current one —
    # each step's own *.status.json persists on disk after that step
    # finishes, so completed steps still show their final count here.
    for step in ("fetch", "classify", "generate"):
        step_label, step_status_file = STAGE_LABELS[step]
        if step_status_file and (run_dir / f"{step_status_file}.status.json").exists():
            render_step_progress(run_dir, step_status_file, step_label)

    with st.expander("Full log", expanded=(stage not in ("done", "failed"))):
        log_path = run_dir / "pipeline.log"
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            st.code(text[-8000:] or "(empty)", language="text")
        else:
            st.caption("(no log yet)")

    summary_path = run_dir / "products_6_images.xlsx"
    final_path = run_dir / "final_output.xlsx"
    if stage == "done" and summary_path.exists():
        st.divider()
        st.subheader("Result")
        with open(summary_path, "rb") as f:
            st.download_button(
                "Download products_6_images.xlsx", f,
                file_name="products_6_images.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{run_id}",
            )
        try:
            final_df = pd.read_excel(final_path)
            st.write("Per-slot status breakdown:")
            st.dataframe(final_df["Status"].value_counts().rename("count"))
            needs_review = final_df[final_df["Status"].astype(str).str.contains("needs review", case=False)]
            if len(needs_review):
                st.warning(f"{len(needs_review)} slot(s) flagged \"needs review\" — "
                          "these are on disk and linked, just never passed automatic verification.")
                st.dataframe(needs_review[["Product_ID", "SKU", "Slot", "Image_Type", "Status"]])
        except Exception as e:
            st.caption(f"(Couldn't load preview: {e})")

    # Auto-refresh while still active — reruns this whole script every
    # ~3s so progress updates without the user touching anything, driven
    # by re-reading disk state each time rather than any in-memory
    # session data, which is what makes this survive a fresh page load.
    if stage not in ("done", "failed"):
        time.sleep(3)
        st.rerun()


st.title("OZi Toys Image Pipeline")
st.caption(
    "Upload a product list, run the full pipeline, and watch every step — "
    "progress, live logs, and errors. Runs happen in the background on the "
    "server, independent of this browser tab — closing it, losing your "
    "connection, or reloading the page will NOT stop or lose a run; just "
    "come back and pick it from the list below."
)

with st.sidebar:
    st.subheader("Environment check")
    checks = env_check()
    all_ok = all(v for k, v in checks.items() if "optional" not in k)
    for name, ok in checks.items():
        st.write(("✅ " if ok else "❌ ") + name)
    if not all_ok:
        st.warning("Missing required credentials — the run will fail immediately. "
                   "Check your .env file.")
    st.divider()
    workers = st.number_input(
        "Workers (concurrency)", min_value=1, max_value=32, value=8,
        help="How many products/images are processed in parallel. Higher = faster "
             "but more likely to hit API rate limits.",
    )
    overwrite = st.checkbox(
        "Force-regenerate existing images (--overwrite)", value=False,
        help="Off by default so re-running never pays for the same image twice.",
    )
    model_override = st.text_input(
        "Image model override (optional)", value=os.environ.get("GEMINI_IMAGE_MODEL", ""),
        help="Leave blank to use the script's default.",
    )
    if model_override:
        os.environ["GEMINI_IMAGE_MODEL"] = model_override

tab_new, tab_monitor = st.tabs(["Start new run", "Watch a run"])

with tab_new:
    uploaded = st.file_uploader(
        "Upload product list (.xlsx or .csv — needs a 'Product ID' or 'Admin Panel Link' column)",
        type=["xlsx", "csv"],
    )
    run_clicked = st.button("Run pipeline", type="primary", disabled=uploaded is None)

    if run_clicked and uploaded is not None:
        if not all_ok:
            st.error("Fix the missing environment variables in the sidebar before running.")
            st.stop()

        # uuid suffix, not just the timestamp — on a shared instance, two
        # people (or one impatient double-click) starting a run in the
        # same second would otherwise get the SAME folder and silently
        # clobber each other's input/output files mid-run.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / f"input{Path(uploaded.name).suffix}"
        input_path.write_bytes(uploaded.getvalue())

        launch_run(input_path, run_dir, workers, overwrite)
        st.session_state["just_started_run_id"] = run_id
        st.success(f"Started run `{run_id}` in the background — switch to "
                  "the \"Watch a run\" tab to follow it (it'll be selected "
                  "there automatically).")

with tab_monitor:
    runs = list_runs()
    if not runs:
        st.caption("No runs yet — start one from the \"Start new run\" tab.")
    else:
        run_ids = [r[0] for r in runs]
        default_idx = 0
        just_started = st.session_state.get("just_started_run_id")
        if just_started in run_ids:
            default_idx = run_ids.index(just_started)

        def _format_run(rid: str) -> str:
            stage = dict(runs)[rid].get("stage", "unknown")
            marker = {"done": "✅", "failed": "❌"}.get(stage, "⏳")
            return f"{marker} {rid} ({stage})"

        selected = st.selectbox(
            "Run", run_ids, index=default_idx, format_func=_format_run,
            help="Every run ever started, newest first — including ones from "
                 "before this browser tab existed.",
        )
        if st.button("Refresh now"):
            st.rerun()
        render_run(selected)
