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
#
# product_matches/issues_found force the model to itemize specific checks
# (product/variant identity, completeness, parts, packaging, logos/text)
# BEFORE it commits to a slot number — the same "enumerate before verdict"
# pattern used in verify_dimension_image (a single holistic "does this fit
# the slot?" judgment let through images that fit the slot's generic
# description while still being the wrong product or missing parts).
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "product_matches": {
            "type": "BOOLEAN",
            "description": ("True only if this photo clearly shows the SAME product "
                           "described above — same shape, color, parts/accessories, and "
                           "variant. False if it shows a different product, a different "
                           "color or variant, or the product is not clearly identifiable "
                           "in the photo."),
        },
        "issues_found": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": ("List any specific problems with this photo, checked one by "
                           "one: wrong product or variant; damaged, incomplete, or "
                           "deformed product; wrong color; missing or extra parts/"
                           "accessories; wrong or clearly outdated packaging; blurry; "
                           "watermarked; wrong background/framing for this slot type; "
                           "wrong, unreadable, or mismatched logos or printed text. Empty "
                           "list if none of these apply."),
        },
        "slot": {"type": "INTEGER"},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
        # The fields below answer a DIFFERENT question from product_matches/
        # slot: "is this specific photo good raw MATERIAL to feed into image
        # generation as a reference?" — correct product identity does not
        # imply that. A photo can genuinely be product_matches=true, slot=2
        # (fits fine as this category's own "Angle" deliverable) and still
        # be a poor GENERATION reference for a DIFFERENT slot — e.g. too
        # zoomed in, the product half out of frame, or packaging covering
        # most of the product. Without this, every covered photo was
        # treated as equally trustworthy raw material regardless of how
        # little of the product it actually shows.
        "product_visibility": {
            "type": "NUMBER",
            "description": ("0.0-1.0: how much of the product's overall shape/surface is "
                           "actually visible and unobstructed in this photo (not overlaid "
                           "by packaging, other objects, a hand, or cropped out of frame). "
                           "1.0 = the whole product is clearly visible."),
        },
        "image_quality": {
            "type": "NUMBER",
            "description": ("0.0-1.0: sharpness/lighting/resolution quality of this photo "
                           "as a piece of source material, independent of composition — "
                           "1.0 = sharp, well-lit, high-resolution."),
        },
        "full_product_visible": {
            "type": "BOOLEAN",
            "description": "True only if the ENTIRE product (not just part of it) is within the frame, uncropped.",
        },
        "packaging_present": {
            "type": "BOOLEAN",
            "description": "True if the product is shown inside/behind its retail packaging (box, blister pack) rather than bare.",
        },
        "view_angle": {
            "type": "STRING",
            "enum": ["front", "front_3q", "side", "back", "top", "angle_other"],
            "description": "Which camera angle this photo was taken from.",
        },
        "usable_as_reference": {
            "type": "BOOLEAN",
            "description": ("True only if this specific photo is good enough to hand to "
                           "an image-generation model as reference material for OTHER "
                           "slots (not just to satisfy its own slot) — i.e. "
                           "product_visibility and image_quality are both reasonably high "
                           "and full_product_visible is true. A photo can still be "
                           "usable_as_reference=false even when product_matches=true and "
                           "slot > 0 (it's fine as ITS OWN deliverable but poor raw "
                           "material for generating a different slot from)."),
        },
        "reason": {"type": "STRING"},
    },
    "required": ["product_matches", "issues_found", "slot", "confidence",
                "product_visibility", "image_quality", "full_product_visible",
                "packaging_present", "view_angle", "usable_as_reference", "reason"],
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

First check product_matches and issues_found (see their field descriptions) —
go through each possible problem one by one rather than judging the photo
holistically at a glance. Only after that, decide which ONE slot number this
image best fulfils.

If product_matches is false, or issues_found is non-empty, or the image
otherwise clearly does not fulfil any of these slots adequately (too blurry,
wrong background for that slot type, product not clearly visible), respond
with slot number 0 regardless of how well the photo's content or composition
happens to otherwise match a slot's generic description.

Separately from the slot decision, also assess this photo as potential
GENERATION REFERENCE MATERIAL for a DIFFERENT slot than the one it covers —
this is a different question from "does it fulfil ITS OWN slot": a photo can
correctly be accepted for its own slot while still being poor raw material to
build another image FROM (too zoomed in, product partly out of frame,
packaging covering most of the product, low quality). Fill in
product_visibility, image_quality, full_product_visible, packaging_present,
view_angle, and usable_as_reference honestly and independently of the slot
decision above — do not default usable_as_reference to true just because the
image was accepted for a slot.

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

    # Deterministic override — don't trust the model's own combined "slot"
    # pick over its own itemized findings. A real image scored well against
    # a slot's generic wording (right composition, right background) while
    # product_matches was actually false or issues_found had real entries;
    # forcing slot=0 here whenever either fires means a stray "slot: 3"
    # can never smuggle a flagged image past classification into
    # Covered_Slots, same non-bypassable pattern as the axis-swap checks in
    # verify_dimension_image.
    issues = parsed.get("issues_found") or []
    if not parsed.get("product_matches", True) or issues:
        parsed["slot"] = 0
        base_reason = parsed.get("reason", "")
        if not parsed.get("product_matches", True):
            parsed["reason"] = f"product_mismatch: {base_reason}"
        elif issues:
            parsed["reason"] = f"issues_found {issues}: {base_reason}"

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
                covered[slot_num] = {
                    "image_url": url,
                    "confidence": result.get("confidence", "low"),
                    "reason": result.get("reason", ""),
                    # Reference-suitability signals — a SEPARATE question
                    # from "does this fulfil its own slot" (see
                    # RESPONSE_SCHEMA/build_classification_prompt). Consumed
                    # by generate_missing_images_gcp.py's build_reference_set
                    # to decide whether this photo should also be offered as
                    # generation material for OTHER slots, not just kept as
                    # its own slot's deliverable.
                    "usable_as_reference": result.get("usable_as_reference", False),
                    "image_quality": result.get("image_quality", 0.0),
                    "product_visibility": result.get("product_visibility", 0.0),
                    "full_product_visible": result.get("full_product_visible", False),
                    "packaging_present": result.get("packaging_present", False),
                    "view_angle": result.get("view_angle", ""),
                }
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
               "Missing_Slots": "", "Covered_Slots": "", "Unmatched_Images": "",
               "Reference_Quality": ""}

    result = classify_product(rules, row, model, project_id, region, tokens)

    if result["status"] == "no_rule_for_category":
        return {**base, "Rule_Category": "", "Status": "no_rule_for_category",
               "Missing_Slots": "", "Covered_Slots": "", "Unmatched_Images": "",
               "Reference_Quality": "", "_unknown_category": row.get("Category_L1", "") or "(blank)"}

    return {
        **base,
        "Rule_Category": result["rule_category"],
        "Status": result["status"],
        "Missing_Slots": "; ".join(f"{m['slot']}:{m['image_type']}"
                                   for m in result["missing_slots"]),
        "Covered_Slots": format_slot_map(
            {slot: info["image_url"] for slot, info in result["covered_slots"].items()}),
        "Unmatched_Images": "; ".join(u["image_url"] for u in result["unmatched_images"]),
        # JSON blob keyed by slot number — the reference-suitability signals
        # (see classify_product) for whichever image ended up covering that
        # slot. A plain "slot:url" string (Covered_Slots) has no room for
        # this; consumed by generate_missing_images_gcp.py's
        # build_reference_set to decide whether a covered photo should also
        # be offered as generation material for OTHER slots.
        "Reference_Quality": json.dumps({
            str(slot): {k: v for k, v in info.items() if k != "image_url"}
            for slot, info in result["covered_slots"].items()
        }),
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
