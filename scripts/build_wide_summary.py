"""
Step 4 (optional): Pivot final_output.xlsx into one-row-per-product
=======================================================================
generate_missing_images_gcp.py writes one row per product PER SLOT (long
format) — useful for auditing status per image, but not what you want to
hand to someone who just wants "the 6 images for this product" at a
glance. This pivots that into one row per product with 6 image-link
columns, each a real clickable hyperlink (not just URL-shaped text).

Usage:
    python3 build_wide_summary.py --input final_output.xlsx --out products_6_images.xlsx
"""
import argparse

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from pipeline_lib import write_link_cell

FONT_NAME = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F2937")
HEADER_FONT = Font(name=FONT_NAME, size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name=FONT_NAME, size=10)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def load_long_format(path: str) -> tuple:
    """Returns (products dict keyed by Product_ID, ordered list of Product_IDs)."""
    src = openpyxl.load_workbook(path).active
    rows = list(src.iter_rows(min_row=2, values_only=True))

    products = {}
    order = []
    for (pid, sku, name, category, slot, image_type, status, image_source, gcp_link) in rows:
        if pid not in products:
            products[pid] = {"sku": sku, "name": name, "category": category,
                             "slots": {}, "note": ""}
            order.append(pid)

        if status == "no_rule_for_category":
            products[pid]["note"] = (f"No rule for category {category!r} — cannot "
                                     f"determine required images")
            continue

        if status in ("Existing", "Existing (quality_check_failed)"):
            # image_source is still the original admin-panel URL in both
            # cases — quality_check_failed just means we couldn't verify
            # its resolution, not that it was replaced with anything.
            link, label = image_source, image_source
        elif status in ("Generated", "Existing (enhanced)"):
            if gcp_link:
                link, label = gcp_link, gcp_link
            else:
                link, label = None, f"{image_source} (LOCAL FILE — not uploaded to GCS)"
        elif status in ("Generated (needs review)", "MANUAL_REVIEW_REQUIRED"):
            # HARD GATE — an image that exhausted every verify-and-retry
            # attempt without a clean pass ("Generated (needs review)"), or
            # was routed to manual review before generation was even
            # attempted at all (ambiguous/missing dimension data — see
            # classify_dimensions in pipeline_lib.py), must never appear as
            # a normal, clickable image link in the actual deliverable.
            # Previously this branch still wrote a live gcp_link here,
            # distinguished only by a "(NEEDS REVIEW)" text suffix — nothing
            # stopped someone clicking it and treating it as a passing
            # image. No link, ever, for either status — same as a genuine
            # generation failure below.
            link, label = None, f"[{status}]"
            note_flag = f"Slot {slot} ({image_type}) needs manual review"
            products[pid]["note"] = (products[pid]["note"] + "; " + note_flag
                                     if products[pid]["note"] else note_flag)
        else:
            link, label = None, f"[{status}]"

        products[pid]["slots"][slot] = (label, link)

    return products, order


def write_wide_excel(products: dict, order: list, out_path: str):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Products"
    ws.sheet_view.showGridLines = False

    headers = ["Product_ID", "SKU", "Name", "Image 1", "Image 2", "Image 3",
               "Image 4", "Image 5", "Image 6", "Notes"]
    widths = [11, 16, 40, 32, 32, 32, 32, 32, 32, 40]
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=i, value=h)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = WRAP
        c.border = BORDER
        ws.column_dimensions[get_column_letter(i)].width = widths[i - 1]
    ws.freeze_panes = "A2"

    for r_idx, pid in enumerate(order, start=2):
        p = products[pid]
        ws.cell(row=r_idx, column=1, value=pid).font = BODY_FONT
        ws.cell(row=r_idx, column=2, value=p["sku"]).font = BODY_FONT
        ws.cell(row=r_idx, column=3, value=p["name"]).font = BODY_FONT
        for col in (1, 2, 3):
            ws.cell(row=r_idx, column=col).alignment = WRAP
            ws.cell(row=r_idx, column=col).border = BORDER

        for slot_num in range(1, 7):
            label, link = p["slots"].get(slot_num, ("", None))
            write_link_cell(ws, r_idx, 3 + slot_num, label, url=link,
                            font=BODY_FONT, border=BORDER)

        note_cell = ws.cell(row=r_idx, column=10, value=p["note"])
        note_cell.font = BODY_FONT
        note_cell.alignment = WRAP
        note_cell.border = BORDER
        ws.row_dimensions[r_idx].height = 40

    wb.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="final_output.xlsx from step 3")
    ap.add_argument("--out", default="products_6_images.xlsx")
    args = ap.parse_args()

    products, order = load_long_format(args.input)
    write_wide_excel(products, order, args.out)
    print(f"Done -> {args.out} ({len(order)} products)")


if __name__ == "__main__":
    main()
