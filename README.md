# OZi Toys — Bulk Product Image Pipeline

Automatically checks which of the 6 required catalog images each Toy
product already has, and generates the missing ones — while preserving
the actual product (no redesigning the toy).

```
products_input.xlsx (Product ID or admin panel link)
        │
        ▼
1. scripts/fetch_product_details.py    -> pulls images, category, description from admin API
        │
        ▼
2. scripts/classify_images.py           -> Gemini vision (Vertex AI) checks which of the 6 slots each existing image fulfils
        │
        ▼
3. scripts/generate_missing_images_gcp.py -> generates only the missing slots (image-to-image, product preserved)
        │
        ▼
final_output.xlsx  (Product_ID, SKU, Slot, Status, Image_Source, GCP_Link)
```

Steps 2 and 3 both run on Vertex AI — one GCP credential covers both, no
separate Anthropic account needed. `GCP_Link` is filled in automatically if
you configure GCS auto-upload (see below); otherwise it's left blank for
you to fill in after a manual upload.

Rules for what each of the 6 images should show, per category (16 Toy
categories), live in `rules/toys_rule_master.json` — edit that file to
change the rules, no code changes needed. To edit the human-readable
`rules/Toys_Rule_Master.xlsx` sheet instead, edit it and run
`python3 scripts/build_toys_rule_master.py --from xlsx` to regenerate the
JSON (or `--from json` to go the other way after editing the JSON by hand).
Category matching is tolerant of case, spacing, and `&`/`and`, and falls
back from `Category_L1` to `Category_L2`/`Category_L3` if the top-level
category isn't in the rule master.

---

## 0. Setup

```bash
cd ozi_toys_image_pipeline
pip install -r requirements.txt
cp .env.example .env
# edit .env with your real credentials, then:
set -a; source .env; set +a
```

Use `set -a; source .env; set +a`, not `export $(grep -v '^#' .env | xargs)` —
the latter silently truncates any value containing a space (e.g.
`GCS_PREFIX="toys test"` becomes just `toys`) because `xargs` word-splits on
whitespace before `export` ever sees it. Quote any `.env` value that
contains a space (`GCS_PREFIX="toys test"`); `source` handles the quotes
correctly, `export $(... | xargs)` does not, even quoted.

**Never** paste your tokens/keys into chat or commit `.env` anywhere.

You need two credentials (see `.env.example`):
| Variable | What it's for | Where to get it |
|---|---|---|
| `OZI_API_KEY` | Reads product data from your admin panel | Issued as a long-lived API key (doesn't expire in ~24h like the old DevTools session token did — no more re-copying it daily) |
| `GOOGLE_APPLICATION_CREDENTIALS` + `GCP_PROJECT_ID` | Classifies existing images (step 2, Gemini vision) AND generates the missing ones (step 3, Gemini image editing) | GCP Console → IAM & Admin → Service Accounts → create one with the **Vertex AI User** role → download its JSON key |

Optionally, to have generated images auto-uploaded to GCS instead of doing
it by hand: `GCS_BUCKET`, `GCS_CREDENTIALS` (can be the same key as above if
that service account also has a Storage role), and optional `GCS_PREFIX` —
see the comments in `.env.example`.

---

## 1. Try it in 2 minutes with the demo product (no OZI token needed)

This creates one fake "Soft Toy" product with a synthetic photo, so you
can see the classify + generate steps work before touching real data.

```bash
cd demo
python3 create_demo_data.py
# -> writes demo_soft_toy_front.png + products_detail_demo.csv

python3 ../scripts/classify_images.py \
    --input products_detail_demo.csv \
    --rules ../rules/toys_rule_master.json \
    --out classification_result_demo.csv

python3 ../scripts/generate_missing_images_gcp.py \
    --products products_detail_demo.csv \
    --classification classification_result_demo.csv \
    --rules ../rules/toys_rule_master.json \
    --image_out_dir generated_images_demo \
    --out final_output_demo.xlsx
```

Open `demo/final_output_demo.xlsx` — you should see slot 1 (Front) marked
`Existing`, and the other 5 slots marked `Generated`, with new images sitting
in `demo/generated_images_demo/`.

(Only the GCP credentials are needed for the demo — it skips the
admin-panel fetch step entirely.)

---

## 2. Run it on your real products

1. Fill in `sample_input/products_input_sample.xlsx` (or make your own)
   with a `Product ID` column (or an `Admin Panel Link` column — the
   product ID is auto-extracted from the `?id=` part of the URL).

2. Run all three steps:

```bash
python3 scripts/fetch_product_details.py \
    --input sample_input/products_input_sample.xlsx \
    --out products_detail.csv

python3 scripts/classify_images.py \
    --input products_detail.csv \
    --rules rules/toys_rule_master.json \
    --out classification_result.csv

python3 scripts/generate_missing_images_gcp.py \
    --products products_detail.csv \
    --classification classification_result.csv \
    --rules rules/toys_rule_master.json \
    --image_out_dir generated_images \
    --out final_output.xlsx
```

3. Open `final_output.xlsx`. If GCS auto-upload is configured, `GCP_Link`
   is already filled in for every `Generated` row. Otherwise, upload the
   file from `generated_images/` to your GCS bucket yourself and paste the
   resulting link into the `GCP_Link` column.

Useful flags for the rollout below:
- `classify_images.py --limit N` / `generate_missing_images_gcp.py --limit N`
  — only process the first N products (use for the 5–10/50/500 stages).
- `generate_missing_images_gcp.py` skips regenerating a slot whose output
  file already exists in `--image_out_dir` (so a re-run after a crash
  doesn't re-bill everything already done); pass `--overwrite` to force
  regeneration. Already-uploaded GCS links are cached the same way (see
  `--image_out_dir/.gcs_uploads.json`) so a re-run doesn't re-upload them.
- `classify_images.py --model gemini-2.5-pro` to swap the classification
  model (default `gemini-2.5-flash`, or set `GEMINI_CLASSIFY_MODEL`).

### Bulk runs (100s–2000 products): concurrency, resuming, monitoring

All three scripts take `--workers N` (default 8) — how many products/slots
are processed at once. This is also the effective rate-limit control
against your Vertex AI / admin-API quota: every call already retries
429/500 with backoff, but setting `--workers` far above what your quota
actually allows just means most requests spend their time retrying
instead of making progress. Start at 8 and raise it gradually while
watching for repeated retry messages.

**Resuming after a crash or Ctrl-C:** every script writes
`<out-file>.checkpoint.jsonl` next to its `--out` file, recording each
completed unit of work (one product for fetch/classify, one product+slot
for generate) as soon as it finishes. Re-running the exact same command
(same `--out`) picks up where it left off — already-done work is skipped
entirely, not re-billed or re-asked. Safe to interrupt at any time; you
only ever lose whatever was still in flight at that instant.

**Monitoring an unattended run:** each script also writes
`<out-file>.status.json`, updated every ~10 completions, with `total`,
`done`, per-status counts, and `started_at`/`updated_at` timestamps. On a
VM with nobody watching the terminal, `cat <out-file>.status.json` over
SSH (or a cron job that checks `updated_at` isn't going stale) tells you
whether the run is progressing without tailing logs.

Both the `.checkpoint.jsonl` and `.status.json` files are safe to delete
once a run has fully completed and you're happy with the output — they're
only useful for resuming/monitoring that specific run.

---

## Recommended rollout (don't run all 2000 products on day one)

1. **5–10 products** — sanity-check the fetch/classify/generate steps
   actually work with your real credentials and check image quality.
2. **~50 products** — check the rule-matching is right across a few
   different categories.
3. **~500 products** — start noticing edge cases (products with 0 images,
   wrong category, low-quality existing photos).
4. **Full 2000** — once error rate is acceptably low.

## Notes / known limitations

- The current 16-category rule set covers **Toys only**
  (`rules/toys_rule_master.json`). Fashion/other modules from your admin
  panel aren't covered and will show `no_rule_for_category` if you feed
  them in — add categories to the JSON to extend. Run
  `scripts/build_toys_rule_master.py` after editing the JSON or the xlsx
  to check the new category has exactly 6 correctly numbered slots.
- Image generation isn't a 100% preservation guarantee — always spot-check
  generated images before publishing to the live catalog.
- Vision classification + generation calls have a real per-call cost;
  the delay/retry settings in each script are tuned to be polite to your
  admin API, not to minimize your AI spend — budget accordingly at scale.
  Use `--limit` on steps 2 and 3 to cap a run to the rollout stage you're on.
- If Google changes model availability again: `GEMINI_CLASSIFY_MODEL` (step
  2) and `GEMINI_IMAGE_MODEL` (step 3) env vars, or `--model` on
  `classify_images.py`, override the defaults without touching code.
- An invalid `OZI_API_KEY` (HTTP 401) stops `fetch_product_details.py`
  immediately instead of marking every remaining product as failed; a
  missing Vertex AI role or unavailable model does the same in
  `classify_images.py` (HTTP 403/404), writing out whatever was already
  classified first.
- `products_detail.csv` gains an `Input_Ref` column (what was actually in
  the input sheet for that row) and `classification_result.csv` gains a
  `Rule_Category` column (the rule-master category that actually matched,
  which may differ from `Category_L1` if the match fell back to L2/L3).
