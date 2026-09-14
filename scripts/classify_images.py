"""
Step 2: Classify Existing Images Against Toys Rule Master
============================================================
For each product (from products_detail.csv, produced by fetch_product_details.py),
this script:
  1. Looks up the category's required 6 image slots from toys_rule_master.json
  2. Sends each existing product image to Vertex AI Gemini (vision) and asks
     which slot, if any, it fulfils
  3. Outputs a per-product slot map: which slots are COVERED (with the matching
     image URL) and which are MISSING (need generation)

Uses the SAME GCP credentials as step 3 (generate_missing_images_gcp.py) —
no separate account needed for this step.

Requires: GOOGLE_APPLICATION_CREDENTIALS + GCP_PROJECT_ID (never hardcode
the key content — only the file path goes in the env var).

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
    export GCP_PROJECT_ID="your-project-id"
    python3 classify_images.py --input products_detail.csv \
        --rules toys_rule_master.json \
        --out classification_result.csv

Output columns:
  Product_ID, SKU, Category_L1, Rule_Category (the rule-master category that
  actually matched), Status, Missing_Slots, Covered_Slots, Unmatched_Images
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from io import BytesIO

import pandas as pd
import requests
from PIL import Image

from pipeline_lib import (
    Checkpoint,
    RuleMaster,
    VertexTokenProvider,
    format_slot_map,
    get_image_bytes,
    run_concurrent,
    split_image_urls,
    write_status,
)

# Flash is the cost/quality sweet spot for a high-volume classifier that
# only has to pick a slot number. Override with --model, or GEMINI_CLASSIFY_MODEL,
# if your project's Model Garden doesn't have this one (or you want Pro instead).
DEFAULT_MODEL = os.environ.get("GEMINI_CLASSIFY_MODEL", "gemini-2.5-flash")
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
# Worth another attempt: throttling, transient backend faults, a stale token
# (refreshed once, then retried). 403/404 mean the whole run is misconfigured
# (missing role, wrong project/region/model) — those are fatal, not per-image.
RETRYABLE_STATUS = {401, 408, 429, 500, 502, 503, 504}

# Constrains the response to valid JSON in this exact shape, so there is no
# markdown fence or preamble to strip off before parsing. This is Gemini's
# response_schema / responseMimeType structured-output mechanism.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "slot": {"type": "INTEGER"},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
        "reason": {"type": "STRING"},
    },
    "required": ["slot", "confidence", "reason"],
}

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

# Content matching alone let through a genuinely low-quality existing photo
# (251x251px, when every other real photo on the same product was
# 950-1500px) — Gemini judged it "yes this is the front view" without any
# sense that its resolution was far below catalog-usable. This floor
# rejects a candidate on quality before it ever reaches that judgement,
# so a slot like that gets regenerated instead of silently accepted.
MIN_EXISTING_IMAGE_PX = 500


class FatalClassifyError(Exception):
    """A failure that will affect every remaining row (missing Vertex AI
    role, wrong project/region, unknown model) — abort rather than mark
    2000 products as unclassified one API call at a time."""


def build_classification_prompt(slots: list, product_name: str = "", description: str = "",
                               specifications: str = "") -> str:
    slot_list = "\n".join(
        f"{s['slot']}. {s['image_type']} — {s['description']}" for s in slots
    )
    # Without the product's own name/description, the model was judging
    # images purely on generic slot wording shared by the whole category
    # (e.g. "Cars & RC Toys" slots mention a "controller") and had no way
    # to know THIS product doesn't have one — that caused it to both
    # mis-accept/reject images and, on the generation side, invent parts
    # that don't exist for this specific product. Giving it the real
    # product context here lets it judge existing images against what the
    # product actually is, not just the category template. Specifications
    # is a separate, more reliable structured field (Brand, Dimensions,
    # Battery Operated, ...) that can disagree with Description's own
    # version of the same facts — labelled as authoritative on conflict.
    product_context = ""
    if product_name or description or specifications:
        spec_block = (f"\nSpecification section (authoritative if it "
                     f"disagrees with the description above):\n{specifications}\n"
                     if specifications else "")
        product_context = (
            f"\nThe product being photographed is: \"{product_name}\"\n"
            f"Product description (use this to understand what the product "
            f"actually is/has — e.g. don't expect a controller or battery "
            f"in the photo if it says the product has none):\n"
            f"{description}\n"
            f"{spec_block}"
        )
    return f"""You are checking an ecommerce product photo against a fixed list of required image slots for its category.
{product_context}
Required slots:
{slot_list}

Look at the attached image and decide which ONE slot number it best fulfils.
If it clearly does not fulfil any of these slots adequately (wrong content, too
blurry, watermarked, wrong background for that slot type, shows a different
product than the one described above, or the product is not clearly visible),
respond with slot number 0.

Give a one short phrase reason."""


def classify_one_image(image_url: str, slots: list, model: str, project_id: str,
                       region: str, tokens: VertexTokenProvider,
                       product_name: str = "", description: str = "",
                       specifications: str = "") -> dict:
    try:
        content, media_type = get_image_bytes(image_url)
    except (requests.RequestException, OSError) as e:
        return {"slot": 0, "confidence": "low", "reason": f"download_failed: {e}"}

    try:
        width, height = Image.open(BytesIO(content)).size
        if min(width, height) < MIN_EXISTING_IMAGE_PX:
            return {"slot": 0, "confidence": "low",
                   "reason": f"resolution_too_low: {width}x{height}px "
                             f"(minimum {MIN_EXISTING_IMAGE_PX}px)"}
    except Exception:
        pass  # not a decodable image at all — let the vision call surface that

    b64 = base64.standard_b64encode(content).decode()
    prompt = build_classification_prompt(slots, product_name, description, specifications)

    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{model}:generateContent"
    )
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": media_type, "data": b64}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        headers = {"Authorization": f"Bearer {tokens.token()}",
                   "Content-Type": "application/json"}
        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            continue

        if resp.status_code == 200:
            return _parse_classify_response(resp.json())

        # Misconfiguration that will fail identically for every remaining
        # image — stop the whole run instead of silently losing every row.
        if resp.status_code in (403, 404):
            raise FatalClassifyError(
                f"HTTP {resp.status_code} from Vertex AI — check that the service "
                f"account has the Vertex AI User role, GCP_PROJECT_ID/GCP_REGION are "
                f"correct, and model {model!r} is available in that project/region. "
                f"Response: {resp.text[:300]}"
            )

        if resp.status_code == 401:
            tokens.token(force_refresh=True)

        if resp.status_code not in RETRYABLE_STATUS:
            # Specific to this one image (bad/unsupported image data, etc.)
            return {"slot": 0, "confidence": "low",
                    "reason": f"rejected_by_api HTTP {resp.status_code}: {resp.text[:200]}"}

        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        if attempt < MAX_RETRIES:
            time.sleep(2 * attempt)

    return {"slot": 0, "confidence": "low", "reason": f"api_error_after_retries: {last_error}"}


def _parse_classify_response(result: dict) -> dict:
    candidates = result.get("candidates") or []
    if not candidates:
        reason = result.get("promptFeedback", {}).get("blockReason", "unknown")
        return {"slot": 0, "confidence": "low", "reason": f"blocked_by_safety_filter: {reason}"}

    parts = candidates[0].get("content", {}).get("parts") or []
    text = next((p["text"] for p in parts if "text" in p), None)
    if text is None:
        finish = candidates[0].get("finishReason", "no reason given")
        return {"slot": 0, "confidence": "low", "reason": f"no_text_in_response: {finish}"}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return {"slot": 0, "confidence": "low", "reason": f"unparsable_response: {e}"}

    return parsed


def classify_product(rules: RuleMaster, row, model: str, project_id: str,
                     region: str, tokens: VertexTokenProvider) -> dict:
    rule_category, slot_defs = rules.match(
        row.get("Category_L1", ""), row.get("Category_L2", ""), row.get("Category_L3", "")
    )
    if not slot_defs:
        return {"status": "no_rule_for_category", "rule_category": ""}

    slots = [slot_defs[n] for n in sorted(slot_defs)]
    covered = {}
    unmatched = []
    product_name = row.get("Name", "")
    description = row.get("Description", "")
    specifications = row.get("Specifications", "")

    for url in split_image_urls(row.get("Image_URLs", "")):
        result = classify_one_image(url, slots, model, project_id, region, tokens,
                                    product_name, description, specifications)
        slot_num = result.get("slot", 0)
        if slot_num in slot_defs:
            # Keep the highest-confidence image per slot.
            existing = covered.get(slot_num)
            if not existing or (CONFIDENCE_RANK.get(result.get("confidence"), 0)
                                > CONFIDENCE_RANK.get(existing["confidence"], 0)):
                covered[slot_num] = {"image_url": url,
                                     "confidence": result.get("confidence", "low"),
                                     "reason": result.get("reason", "")}
        else:
            unmatched.append({"image_url": url, "reason": result.get("reason", "")})
        time.sleep(REQUEST_DELAY_SECONDS)

    return {
        "status": "ok",
        "rule_category": rule_category,
        "covered_slots": covered,
        "missing_slots": [{"slot": n, "image_type": slot_defs[n]["image_type"]}
                          for n in sorted(slot_defs) if n not in covered],
        "unmatched_images": unmatched,
    }


def process_classify_task(row, rules: RuleMaster, model: str, project_id: str,
                          region: str, tokens: VertexTokenProvider) -> dict:
    """Does the real work for ONE product and returns a classification_result
    row. May raise FatalClassifyError (misconfiguration that would fail
    identically for every remaining product) — the caller decides what to
    do with that; it is NOT swallowed here.
    """
    base = {"Product_ID": row["Product_ID"], "SKU": row["SKU"],
           "Category_L1": row.get("Category_L1", "")}

    if row.get("Fetch_Status") != "ok" or not split_image_urls(row.get("Image_URLs", "")):
        return {**base, "Rule_Category": "", "Status": "skipped_no_images_or_fetch_failed",
               "Missing_Slots": "", "Covered_Slots": "", "Unmatched_Images": ""}

    result = classify_product(rules, row, model, project_id, region, tokens)

    if result["status"] == "no_rule_for_category":
        return {**base, "Rule_Category": "", "Status": "no_rule_for_category",
               "Missing_Slots": "", "Covered_Slots": "", "Unmatched_Images": "",
               "_unknown_category": row.get("Category_L1", "") or "(blank)"}

    return {
        **base,
        "Rule_Category": result["rule_category"],
        "Status": result["status"],
        "Missing_Slots": "; ".join(f"{m['slot']}:{m['image_type']}"
                                   for m in result["missing_slots"]),
        "Covered_Slots": format_slot_map(
            {slot: info["image_url"] for slot, info in result["covered_slots"].items()}),
        "Unmatched_Images": "; ".join(u["image_url"] for u in result["unmatched_images"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="products_detail.csv from fetch step")
    ap.add_argument("--rules", required=True, help="toys_rule_master.json")
    ap.add_argument("--out", default="classification_result.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Gemini model id on Vertex AI (default {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, help="only classify the first N products "
                                              "(use this for the 5-10 product sanity check)")
    ap.add_argument("--workers", type=int, default=8,
                    help="max concurrent products being classified at once (default 8) — "
                         "also the effective rate-limit control against Vertex AI's "
                         "per-minute quota; raise only as far as your quota allows.")
    args = ap.parse_args()

    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        sys.exit("ERROR: set GCP_PROJECT_ID environment variable first.")
    region = os.environ.get("GCP_REGION", "us-central1")
    tokens = VertexTokenProvider(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    rules = RuleMaster.load(args.rules)
    products = pd.read_csv(args.input).fillna("")
    if args.limit:
        products = products.head(args.limit)

    rows = [row for _, row in products.iterrows()]
    total = len(rows)

    # One product = one checkpoint unit. A crash or Ctrl-C mid-run loses at
    # most whichever products were in flight at that instant when re-run
    # with the same --out — everything already classified is skipped, not
    # re-billed or re-asked.
    checkpoint = Checkpoint(args.out + ".checkpoint.jsonl")
    keys = [str(row["Product_ID"]) for row in rows]
    already_done = sum(1 for k in keys if checkpoint.is_done(k))
    if already_done:
        print(f"Resuming from checkpoint: {already_done}/{total} products already classified.")

    progress_lock = threading.Lock()
    progress = {"n": 0}
    unknown_categories = set()
    live_counts = {"ok": 0, "no_rule_for_category": 0, "skipped_no_images_or_fetch_failed": 0}
    status_path = args.out + ".status.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def worker(row):
        key = str(row["Product_ID"])
        if checkpoint.is_done(key):
            return checkpoint.get(key)
        # FatalClassifyError intentionally propagates uncaught — it means
        # the whole run is misconfigured (bad role/project/region/model),
        # not that this one product failed, so it should stop everything
        # rather than get silently recorded as a per-row failure.
        out_row = process_classify_task(row, rules, args.model, project_id, region, tokens)
        checkpoint.record(key, out_row)
        return out_row

    def on_result(_i, _row, out_row):
        with progress_lock:
            progress["n"] += 1
            n = progress["n"]
            live_counts[out_row.get("Status", "ok")] = live_counts.get(
                out_row.get("Status", "ok"), 0) + 1
        if out_row.get("_unknown_category"):
            unknown_categories.add(out_row["_unknown_category"])
        if n % 10 == 0 or n == total:
            print(f"[{n}/{total}] done")
            write_status(status_path, total=total, done=n, started_at=started_at,
                        **live_counts)

    # pd.DataFrame() turns the union of every dict's keys into columns, so
    # the internal-only "_unknown_category" marker (used only to build the
    # end-of-run summary above) has to be stripped before writing — unlike
    # write_final_excel in the generate step, which only reads specific
    # named keys and ignores the rest.
    def strip_internal(row):
        return {k: v for k, v in row.items() if not k.startswith("_")}

    try:
        out_rows = run_concurrent(rows, worker, max_workers=args.workers, on_result=on_result)
    except FatalClassifyError as e:
        checkpoint.close()
        completed = [strip_internal(checkpoint.get(k)) for k in keys if checkpoint.is_done(k)]
        pd.DataFrame(completed).to_csv(args.out, index=False)
        sys.exit(f"\nFATAL: {e}\n"
                 f"Partial results ({len(completed)}/{total} products) written to {args.out}. "
                 f"Fix the issue above, then re-run with the same --out to resume — "
                 f"already-classified products won't be re-billed.")

    checkpoint.close()
    write_status(status_path, total=total, done=total, started_at=started_at,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **live_counts)
    pd.DataFrame([strip_internal(r) for r in out_rows]).to_csv(args.out, index=False)
    print(f"\nDone -> {args.out}")
    if unknown_categories:
        print("\nThese product categories are not in the rule master and were "
              "skipped — add them to the rules JSON to cover them:")
        for name in sorted(unknown_categories):
            print(f"  - {name}")
        print(f"Rule master currently covers: {', '.join(rules.category_names)}")


if __name__ == "__main__":
    main()
