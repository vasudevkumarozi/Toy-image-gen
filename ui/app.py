"""
OZi Toys Image Pipeline — checking UI
=======================================================================
A Streamlit front-end for the 4-step pipeline (fetch -> classify ->
generate -> build_wide_summary), built specifically so you can upload a
product list and WATCH the run happen — live logs, a progress bar per
step (backed by the same status.json files the pipeline already writes
for unattended VM runs), and a clear red stop on the first failed step
instead of silently continuing.

This is meant to be run BOTH locally (to sanity-check the pipeline before
ever touching the deploy instance) and later on the instance itself, so
"does this actually work" has a UI answer, not just a terminal you have
to SSH into and tail.

Run it with:
    streamlit run ui/app.py
"""
# Defers annotation evaluation so `str | None` below doesn't hard-require
# Python 3.10+ — the deploy instance's base image isn't guaranteed to be
# on a version that supports PEP 604 syntax natively.
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RULES_PATH = REPO_ROOT / "rules" / "toys_rule_master.json"
RUNS_DIR = REPO_ROOT / "ui_runs"
RUNS_DIR.mkdir(exist_ok=True)

# Loaded once at import time (Streamlit re-executes this file on every
# interaction, but load_dotenv is cheap and idempotent) — this is what
# lets the UI process see OZI_API_KEY / GCP_PROJECT_ID / etc. without
# needing the `set -a; source .env; set +a` shell dance every script here
# otherwise relies on, since Streamlit isn't launched through that shell.
load_dotenv(REPO_ROOT / ".env")

st.set_page_config(page_title="OZi Toys Image Pipeline", layout="wide")


def run_step(cmd: list, status_json_path: str | None, container) -> tuple:
    """Runs cmd as a subprocess inside `container` (an st.status(...) block),
    streaming stdout live and — if status_json_path is given — polling it
    for a progress bar. Returns (success, full_log_text).

    Log reading happens on a background thread so the progress bar still
    refreshes every ~1s even during a long gap between printed lines (the
    generate step can go 30-70s between "[n/total] done" prints once
    dimension-image retries are involved) — polling only inside a blocking
    readline() loop would freeze the progress bar during exactly those
    gaps.
    """
    log_lines: list = []
    log_lock = threading.Lock()

    process = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=os.environ.copy(),
    )

    def reader():
        assert process.stdout is not None
        for line in process.stdout:
            with log_lock:
                log_lines.append(line.rstrip("\n"))

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    log_placeholder = container.empty()
    progress_placeholder = container.empty()

    while process.poll() is None or reader_thread.is_alive():
        with log_lock:
            text = "\n".join(log_lines[-300:])
        log_placeholder.code(text or "(waiting for output...)", language="text")
        if status_json_path and os.path.exists(status_json_path):
            try:
                with open(status_json_path, encoding="utf-8") as f:
                    data = json.load(f)
                total = data.get("total", 0)
                done = data.get("done", 0)
                if total:
                    counts = {k: v for k, v in data.items()
                             if k not in ("total", "done", "started_at",
                                         "updated_at", "finished_at")}
                    counts_str = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
                    progress_placeholder.progress(
                        min(done / total, 1.0),
                        text=f"{done}/{total} — {counts_str}" if counts_str else f"{done}/{total}",
                    )
            except (json.JSONDecodeError, OSError):
                pass
        if process.poll() is not None and not reader_thread.is_alive():
            break
        time.sleep(1.0)

    returncode = process.wait()
    with log_lock:
        full_log = "\n".join(log_lines)
    return returncode == 0, full_log


def env_check() -> dict:
    """Mirrors what each script would itself refuse to run without —
    surfacing this BEFORE the run starts is the whole point of a
    pre-deployment checking UI: catch a missing credential here, not 40
    minutes into a 2000-product run on the instance."""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return {
        "OZI_API_KEY": bool(os.environ.get("OZI_API_KEY")),
        "GCP_PROJECT_ID": bool(os.environ.get("GCP_PROJECT_ID")),
        "GOOGLE_APPLICATION_CREDENTIALS (file exists)": bool(creds_path) and os.path.exists(creds_path),
        "GCS_BUCKET (optional — enables auto-upload)": bool(os.environ.get("GCS_BUCKET")),
    }


st.title("OZi Toys Image Pipeline")
st.caption(
    "Upload a product list, run the full pipeline, and watch every step — "
    "progress, live logs, and errors — before this ever runs unattended on "
    "an instance."
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
    # people (or one impatient double-click) starting a run in the same
    # second would otherwise get the SAME folder and silently clobber
    # each other's input/output files mid-run.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / f"input{Path(uploaded.name).suffix}"
    input_path.write_bytes(uploaded.getvalue())

    products_csv = run_dir / "products_detail.csv"
    classification_csv = run_dir / "classification_result.csv"
    images_dir = run_dir / "generated_images"
    final_xlsx = run_dir / "final_output.xlsx"
    summary_xlsx = run_dir / "products_6_images.xlsx"

    python = sys.executable
    pipeline_failed = False

    with st.status("Step 1/4 — Fetching product details from the OZi admin API", expanded=True) as s1:
        ok, _ = run_step(
            [python, str(SCRIPTS_DIR / "fetch_product_details.py"),
             "--input", str(input_path), "--out", str(products_csv),
             "--workers", str(workers)],
            str(products_csv) + ".status.json", s1,
        )
        if ok:
            s1.update(label="Step 1/4 — Fetch complete ✅", state="complete")
        else:
            s1.update(label="Step 1/4 — Fetch FAILED ❌ (see log above)", state="error")
            pipeline_failed = True

    if not pipeline_failed:
        with st.status("Step 2/4 — Classifying existing images against the rule master", expanded=True) as s2:
            ok, _ = run_step(
                [python, str(SCRIPTS_DIR / "classify_images.py"),
                 "--input", str(products_csv), "--rules", str(RULES_PATH),
                 "--out", str(classification_csv), "--workers", str(workers)],
                str(classification_csv) + ".status.json", s2,
            )
            if ok:
                s2.update(label="Step 2/4 — Classification complete ✅", state="complete")
            else:
                s2.update(label="Step 2/4 — Classification FAILED ❌ (see log above)", state="error")
                pipeline_failed = True

    if not pipeline_failed:
        with st.status("Step 3/4 — Generating missing images (the slow step)", expanded=True) as s3:
            cmd = [python, str(SCRIPTS_DIR / "generate_missing_images_gcp.py"),
                  "--products", str(products_csv), "--classification", str(classification_csv),
                  "--rules", str(RULES_PATH), "--image_out_dir", str(images_dir),
                  "--out", str(final_xlsx), "--workers", str(workers)]
            if overwrite:
                cmd.append("--overwrite")
            ok, _ = run_step(cmd, str(final_xlsx) + ".status.json", s3)
            if ok:
                s3.update(label="Step 3/4 — Generation complete ✅", state="complete")
            else:
                s3.update(label="Step 3/4 — Generation FAILED ❌ (see log above)", state="error")
                pipeline_failed = True

    if not pipeline_failed:
        with st.status("Step 4/4 — Building the final summary sheet", expanded=True) as s4:
            ok, _ = run_step(
                [python, str(SCRIPTS_DIR / "build_wide_summary.py"),
                 "--input", str(final_xlsx), "--out", str(summary_xlsx)],
                None, s4,
            )
            if ok:
                s4.update(label="Step 4/4 — Summary complete ✅", state="complete")
            else:
                s4.update(label="Step 4/4 — Summary build FAILED ❌ (see log above)", state="error")
                pipeline_failed = True

    if pipeline_failed:
        st.error(
            "Pipeline stopped at the first failed step — nothing downstream ran. "
            "Fix the issue above and click Run again; already-completed rows/images "
            "are checkpointed, so re-running only redoes what actually failed."
        )
    else:
        st.success("Pipeline complete!")
        st.session_state["last_run_dir"] = str(run_dir)
        st.session_state["last_summary_path"] = str(summary_xlsx)
        st.session_state["last_final_path"] = str(final_xlsx)

if "last_summary_path" in st.session_state and os.path.exists(st.session_state["last_summary_path"]):
    st.divider()
    st.subheader("Result")
    summary_path = st.session_state["last_summary_path"]
    final_path = st.session_state["last_final_path"]

    with open(summary_path, "rb") as f:
        st.download_button(
            "Download products_6_images.xlsx", f,
            file_name="products_6_images.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    try:
        final_df = pd.read_excel(final_path)
        st.write("Per-slot status breakdown (this run):")
        st.dataframe(final_df["Status"].value_counts().rename("count"))
        needs_review = final_df[final_df["Status"].astype(str).str.contains("needs review", case=False)]
        if len(needs_review):
            st.warning(f"{len(needs_review)} slot(s) flagged \"needs review\" — "
                      "these are on disk and linked, just never passed automatic verification.")
            st.dataframe(needs_review[["Product_ID", "SKU", "Slot", "Image_Type", "Status"]])
    except Exception as e:
        st.caption(f"(Couldn't load preview: {e})")
