"""
Runs the full 4-step pipeline (fetch -> classify -> generate ->
build_wide_summary) as ONE self-contained process, writing its own
progress to <run_dir>/pipeline_status.json and all subprocess output to
<run_dir>/pipeline.log.

Built specifically to be launched DETACHED from the checking UI (ui/app.py
starts this with start_new_session=True and does not wait on it) so the
run survives regardless of what happens to the browser tab, SSH tunnel,
or Streamlit session that started it — a real run was silently abandoned
after step 2 when the browser session that had been driving each step
sequentially got torn down by a dropped tunnel, even though the actual
step-2 subprocess itself had already finished cleanly. Running the whole
sequence in one independent process, rather than as blocking subprocess
calls inside the UI's own script execution, removes that failure mode
entirely — this process's parent is init, not Streamlit.

The UI polls pipeline_status.json (and the same per-step *.status.json
files each script already writes) to render progress, and can reattach to
this run from a fresh page load at any time since everything it needs is
on disk, not in Streamlit session state.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RULES_PATH = REPO_ROOT / "rules" / "toys_rule_master.json"

STEPS = ("fetch", "classify", "generate", "summary")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(status_path: Path, **fields) -> None:
    data = {}
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    data.update(fields)
    data["updated_at"] = _now()
    status_path.write_text(json.dumps(data, indent=2))


def run_step(cmd: list, log_path: Path, step_name: str) -> bool:
    """Runs cmd, appending its output to log_path with a header/footer so
    the log reads as one continuous transcript across all 4 steps."""
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n{'=' * 70}\n=== STEP: {step_name} — {_now()} ===\n{'=' * 70}\n")
        log_f.flush()
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log_f,
                                stderr=subprocess.STDOUT)
        log_f.write(f"\n--- {step_name} exited with code {result.returncode} ---\n")
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--input", required=True, help="uploaded product list, already saved to disk")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "pipeline_status.json"
    log_path = run_dir / "pipeline.log"

    products_csv = run_dir / "products_detail.csv"
    classification_csv = run_dir / "classification_result.csv"
    images_dir = run_dir / "generated_images"
    final_xlsx = run_dir / "final_output.xlsx"
    summary_xlsx = run_dir / "products_6_images.xlsx"

    write_status(status_path, stage="fetch", stage_num=1, started_at=_now(),
                finished_at=None, error=None)

    python = sys.executable

    ok = run_step(
        [python, str(SCRIPTS_DIR / "fetch_product_details.py"),
         "--input", args.input, "--out", str(products_csv),
         "--workers", str(args.workers)],
        log_path, "fetch",
    )
    if not ok:
        write_status(status_path, stage="failed", error="Step 1/4 (fetch) failed — see pipeline.log")
        return

    write_status(status_path, stage="classify", stage_num=2)
    ok = run_step(
        [python, str(SCRIPTS_DIR / "classify_images.py"),
         "--input", str(products_csv), "--rules", str(RULES_PATH),
         "--out", str(classification_csv), "--workers", str(args.workers)],
        log_path, "classify",
    )
    if not ok:
        write_status(status_path, stage="failed", error="Step 2/4 (classify) failed — see pipeline.log")
        return

    write_status(status_path, stage="generate", stage_num=3)
    cmd = [python, str(SCRIPTS_DIR / "generate_missing_images_gcp.py"),
          "--products", str(products_csv), "--classification", str(classification_csv),
          "--rules", str(RULES_PATH), "--image_out_dir", str(images_dir),
          "--out", str(final_xlsx), "--workers", str(args.workers)]
    if args.overwrite:
        cmd.append("--overwrite")
    ok = run_step(cmd, log_path, "generate")
    if not ok:
        write_status(status_path, stage="failed", error="Step 3/4 (generate) failed — see pipeline.log")
        return

    write_status(status_path, stage="summary", stage_num=4)
    ok = run_step(
        [python, str(SCRIPTS_DIR / "build_wide_summary.py"),
         "--input", str(final_xlsx), "--out", str(summary_xlsx)],
        log_path, "summary",
    )
    if not ok:
        write_status(status_path, stage="failed", error="Step 4/4 (summary) failed — see pipeline.log")
        return

    write_status(status_path, stage="done", finished_at=_now(), error=None)


if __name__ == "__main__":
    main()
