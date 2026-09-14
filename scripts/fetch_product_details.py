"""
Step 1: Fetch Product Details from OZi Admin API
==================================================
Reads a sheet of Product IDs (and/or admin panel links), calls the detail
API for each, and writes a flat CSV with every image URL + category path +
description — ready for the next step (image classification against the
Category Rule Master).

SECURITY: the API key is read from an environment variable, never
hardcoded here and never printed/logged. Set it before running:

    export OZI_API_KEY="your_api_key_here"

Usage:
    python3 fetch_product_details.py --input products_input.xlsx --out products_detail.csv

Input file requirements (xlsx or csv):
  Must have at least ONE of these columns:
    - "Product ID"        -> numeric item id, e.g. 42169
    - "Admin Panel Link"  -> e.g. https://ozi-admin-panel.ozi.in/product_setup_listing/view?id=42169
                              (id is extracted automatically from the ?id= param)
  Both may be present, and they are resolved per row: "Product ID" wins when
  that row has one, otherwise the id is taken from that row's link.

Output columns:
  Product_ID, SKU, Name, Category_L1, Category_L2, Category_L3,
  Description, Specifications (the admin panel's separate "Specification"
  section — Brand, Dimensions (LxBxH), Weight, Battery Operated, etc. —
  treated as more authoritative than any overlapping facts embedded in the
  free-text Description), Image_Count, Image_URLs (semicolon-separated),
  Input_Ref, Fetch_Status
"""
import argparse
import os
import re
import sys
import threading
import time
import json
import pandas as pd
import requests

from pipeline_lib import Checkpoint, run_concurrent, write_status

BASE_URL = "https://ozi-admin-panel-be.ozi.in"
DETAIL_ENDPOINT = BASE_URL + "/api/admin/item/{id}"

# --- Safety knobs for bulk runs (1000-2000 products) ---
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
TIMEOUT_SECONDS = 15


class FatalFetchError(Exception):
    """A wrong/revoked OZI_API_KEY fails every remaining row identically —
    stop the whole run rather than burning through the whole sheet one
    already-doomed request at a time."""


def get_token() -> str:
    # A long-lived API key, not a session JWT — doesn't expire in ~24h like
    # the old admin-panel bearer token did, so no more re-copying it from
    # DevTools every day.
    key = os.environ.get("OZI_API_KEY")
    if not key:
        sys.exit(
            "ERROR: OZI_API_KEY environment variable not set.\n"
            "Run: export OZI_API_KEY=\"<your api key>\"\n"
            "(Never paste the key into this script or into chat.)"
        )
    return key


def extract_id_from_link(link: str):
    if not isinstance(link, str):
        return None
    match = re.search(r"id=(\d+)", link)
    return int(match.group(1)) if match else None


def load_input(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _coerce_id(value):
    """'42169', 42169, 42169.0 -> 42169. Anything else (blank, NaN, text) -> None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def resolve_product_ids(df: pd.DataFrame) -> list:
    """Resolve one product id per input row.

    Resolved PER ROW rather than per column: a sheet commonly has both
    columns filled in sparsely (an id on some rows, only a link on others),
    so falling back row-by-row is what the user actually means. Returns a
    list of (product_id_or_None, input_ref) so failures stay traceable back
    to what was in the sheet.
    """
    has_id_col = "Product ID" in df.columns
    has_link_col = "Admin Panel Link" in df.columns
    if not has_id_col and not has_link_col:
        sys.exit("Input file needs a 'Product ID' or 'Admin Panel Link' column.")

    resolved = []
    for _, row in df.iterrows():
        raw_id = row.get("Product ID") if has_id_col else None
        raw_link = row.get("Admin Panel Link") if has_link_col else None

        product_id = _coerce_id(raw_id)
        if product_id is None and isinstance(raw_link, str):
            product_id = extract_id_from_link(raw_link)

        ref_parts = [str(v).strip() for v in (raw_id, raw_link)
                     if v is not None and str(v).strip() and str(v).strip().lower() != "nan"]
        resolved.append((product_id, " | ".join(ref_parts)))
    return resolved


def fetch_one(product_id: int, api_key: str) -> dict:
    # The admin API takes the API key via X-API-Key, not an Authorization
    # Bearer header (verified against the live API — Bearer returns 401
    # "Invalid or expired token" even with a valid key).
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    url = DETAIL_ENDPOINT.format(id=product_id)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return {"status": "ok", "data": resp.json()}
            if resp.status_code == 401:
                return {"status": "auth_error", "data": None}
            if resp.status_code == 404:
                return {"status": "not_found", "data": None}
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = str(e)
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return {"status": f"failed: {last_error}", "data": None}


def flatten_specification(payload: dict) -> str:
    """The admin panel's "Specification" section (Brand, Dimensions (LxBxH),
    Weight, Battery Operated, etc.) is a SEPARATE structured dict in the API
    response — "specification" (or "specifications", same content) — not
    part of the free-text "description" field, which sometimes has its own,
    less reliable version of the same facts (e.g. a "Size: 4 inch" line
    describing the figure's own scale, while Specification's
    "Dimensions (LxBxH)" is the actual product/box footprint). Formats as
    "Key: Value" lines so it parses with the same field-parser as
    Description's structured tail, with this treated as authoritative."""
    spec = payload.get("specification") or payload.get("specifications") or {}
    if not isinstance(spec, dict):
        return ""
    return "\n".join(f"{k}: {v}" for k, v in spec.items() if v not in (None, ""))


def flatten_product(product_id: int, payload: dict) -> dict:
    if payload is None:
        return {
            "Product_ID": product_id, "SKU": "", "Name": "",
            "Category_L1": "", "Category_L2": "", "Category_L3": "",
            "Description": "", "Specifications": "", "Image_Count": 0, "Image_URLs": "",
        }

    main_image = payload.get("image_full_url", "")
    gallery = payload.get("images_full_url", []) or []
    all_images = ([main_image] if main_image else []) + list(gallery)
    # de-duplicate while preserving order
    seen = set()
    unique_images = []
    for url in all_images:
        if url and url not in seen:
            seen.add(url)
            unique_images.append(url)

    category = payload.get("category") or {}
    sub_category = payload.get("sub_category") or {}
    sub_sub_category = payload.get("sub_sub_category") or {}

    return {
        "Product_ID": product_id,
        "SKU": payload.get("sku", ""),
        "Name": payload.get("name", ""),
        "Category_L1": category.get("name", ""),
        "Category_L2": sub_category.get("name", ""),
        "Category_L3": sub_sub_category.get("name", ""),
        "Description": (payload.get("description") or "").strip(),
        "Specifications": flatten_specification(payload),
        "Image_Count": len(unique_images),
        "Image_URLs": ";".join(unique_images),
    }


def process_fetch_task(task: dict, token: str) -> dict:
    """Does the real work for ONE input row and returns a products_detail
    row. Raises FatalFetchError on an auth rejection — intentionally NOT
    caught here, same as FatalClassifyError in classify_images.py — a bad
    key means every remaining row would fail identically, so the whole
    run should stop rather than record 2000 individual auth_error rows.
    """
    pid, input_ref = task["pid"], task["input_ref"]
    if pid is None:
        return {**flatten_product(None, None), "Input_Ref": input_ref,
               "Fetch_Status": "invalid_id"}
    result = fetch_one(pid, token)
    if result["status"] == "auth_error":
        raise FatalFetchError("the admin API rejected OZI_API_KEY (HTTP 401)")
    record = flatten_product(pid, result["data"])
    record["Input_Ref"] = input_ref
    record["Fetch_Status"] = result["status"]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="products_input.xlsx or .csv")
    ap.add_argument("--out", default="products_detail.csv")
    ap.add_argument("--workers", type=int, default=8,
                    help="max concurrent admin-API requests in flight at once "
                         "(default 8) — raise for a faster bulk fetch, but stay "
                         "polite to the admin API's own capacity.")
    args = ap.parse_args()

    token = get_token()
    df = load_input(args.input)
    resolved = resolve_product_ids(df)
    total = len(resolved)

    # Keyed by row position + id, not just id — several rows can have
    # pid=None (blank/invalid rows) and would otherwise collide on the
    # same checkpoint key.
    tasks = [{"key": f"{i}:{pid}", "pid": pid, "input_ref": input_ref}
            for i, (pid, input_ref) in enumerate(resolved)]

    checkpoint = Checkpoint(args.out + ".checkpoint.jsonl")
    already_done = sum(1 for t in tasks if checkpoint.is_done(t["key"]))
    if already_done:
        print(f"Resuming from checkpoint: {already_done}/{total} rows already fetched.")

    progress_lock = threading.Lock()
    progress = {"n": 0}
    live_counts = {}
    status_path = args.out + ".status.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def worker(task):
        if checkpoint.is_done(task["key"]):
            return checkpoint.get(task["key"])
        row = process_fetch_task(task, token)
        checkpoint.record(task["key"], row)
        return row

    def on_result(_i, _task, row):
        with progress_lock:
            progress["n"] += 1
            n = progress["n"]
            status = row.get("Fetch_Status", "unknown")
            live_counts[status] = live_counts.get(status, 0) + 1
        if n % 10 == 0 or n == total:
            print(f"[{n}/{total}] done")
            write_status(status_path, total=total, done=n, started_at=started_at,
                        **live_counts)

    try:
        rows = run_concurrent(tasks, worker, max_workers=args.workers, on_result=on_result)
    except FatalFetchError as e:
        checkpoint.close()
        completed = [checkpoint.get(t["key"]) for t in tasks if checkpoint.is_done(t["key"])]
        pd.DataFrame(completed).to_csv(args.out, index=False)
        sys.exit(f"\nERROR: {e}.\n"
                 f"Partial results ({len(completed)}/{total}) written to {args.out}. "
                 f"Check the key and re-run with the same --out to resume.")

    checkpoint.close()
    write_status(status_path, total=total, done=total, started_at=started_at,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **live_counts)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    ok = (out_df["Fetch_Status"] == "ok").sum()
    print(f"\nDone. {ok}/{total} fetched successfully -> {args.out}")
    failed = out_df[out_df["Fetch_Status"] != "ok"]
    if len(failed):
        print(f"{len(failed)} products had issues (see Fetch_Status column for details).")
    no_images = out_df[(out_df["Fetch_Status"] == "ok") & (out_df["Image_Count"] == 0)]
    if len(no_images):
        print(f"{len(no_images)} fetched products have 0 images — those cannot be "
              f"used as a generation reference and will be reported as "
              f"failed_no_reference_image in step 3.")


if __name__ == "__main__":
    main()
