"""
Builds / maintains the Toys Rule Master in its two forms:
  1. rules/toys_rule_master.json   -> machine-readable, used by the pipeline
  2. rules/Toys_Rule_Master.xlsx   -> human-editable master sheet

Three directions, so the sheet and the JSON can be kept in sync without
needing the original Cataloging 2.0 CSV export (which is not in this repo):

    # apply edits made in the spreadsheet back to the JSON the scripts read
    python3 build_toys_rule_master.py --from xlsx

    # regenerate the spreadsheet after editing the JSON by hand
    python3 build_toys_rule_master.py --from json

    # first-time import from the raw Cataloging 2.0 CSV export
    python3 build_toys_rule_master.py --from csv --src /path/to/Cataloging_2_0_-_Category_wise_title_rules.csv

The JSON is what the pipeline actually reads, so it is the source of truth;
the xlsx is a convenience view of it.
"""
import argparse
import csv
import json
import os
import re
import sys

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JSON = os.path.join(REPO_ROOT, "rules", "toys_rule_master.json")
DEFAULT_XLSX = os.path.join(REPO_ROOT, "rules", "Toys_Rule_Master.xlsx")

SLOTS_SHEET = "Toys_Rule_Master"
TITLES_SHEET = "Title_Rules"
ALIASES_SHEET = "Category_Aliases"

TITLE_FIELDS = [
    ("title_formatting_rule", "Title Formatting Rule"),
    ("title_structure", "Title Structure"),
    ("include_in_title", "Include In Title"),
    ("avoid_remove", "Avoid / Remove"),
    ("brand_exception", "Brand Exception"),
]

IMAGE_ROW_RE = re.compile(r"^image\s*(\d+)$", re.IGNORECASE)


# ---------------------------------------------------------------------
# Source 1: the raw Cataloging 2.0 CSV export
# ---------------------------------------------------------------------
def parse_source_csv(path: str) -> list:
    """Parse the two sections of the raw export.

    The sections are found by content, not by fixed row offsets: an image
    row is any row whose second column reads "Image <n>". Everything above
    the first such row that names a category is a title-rule row. Hardcoded
    offsets broke as soon as the export gained or lost a row.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = [[(c or "").strip() for c in r] for r in csv.reader(f)]

    def cell(row, idx):
        return row[idx] if idx < len(row) else ""

    # --- Section 2 first: category-wise image types and descriptions ---
    categories = {}
    first_image_row = None
    for line_no, row in enumerate(rows):
        match = IMAGE_ROW_RE.match(cell(row, 1))
        if not match:
            continue
        category = cell(row, 0)
        if not category:
            continue
        if first_image_row is None:
            first_image_row = line_no
        categories.setdefault(category, {"category": category, "title_rule": {},
                                         "image_slots": []})
        categories[category]["image_slots"].append({
            "slot": int(match.group(1)),
            "image_type": cell(row, 2),
            "description": cell(row, 3),
        })

    if not categories:
        sys.exit(f"ERROR: found no 'Image <n>' rows in {path}. Is this the "
                 f"Cataloging 2.0 category-wise export?")

    # --- Section 1: title rules, above the image section ---
    for row in rows[:first_image_row]:
        category = cell(row, 0)
        if category in categories and not categories[category]["title_rule"]:
            categories[category]["title_rule"] = {
                "title_formatting_rule": cell(row, 1),
                "title_structure": cell(row, 2),
                "include_in_title": cell(row, 3),
                "avoid_remove": cell(row, 4),
                "brand_exception": cell(row, 7),
            }

    rule_master = list(categories.values())
    for entry in rule_master:
        entry["image_slots"].sort(key=lambda s: s["slot"])
        entry.setdefault("title_rule", {})
        entry.setdefault("aliases", [])
    return rule_master


# ---------------------------------------------------------------------
# Source 2: the human-editable xlsx
# ---------------------------------------------------------------------
def parse_xlsx(path: str) -> list:
    wb = openpyxl.load_workbook(path, data_only=True)
    if SLOTS_SHEET not in wb.sheetnames:
        sys.exit(f"ERROR: {path} has no {SLOTS_SHEET!r} sheet "
                 f"(found: {', '.join(wb.sheetnames)}).")

    ws = wb[SLOTS_SHEET]
    categories = {}
    for r_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        category, slot, image_type, description = (list(row) + [None] * 4)[:4]
        if not category or slot is None:
            continue
        try:
            slot_num = int(slot)
        except (TypeError, ValueError):
            sys.exit(f"ERROR: {path} {SLOTS_SHEET}!B{r_idx}: slot {slot!r} is not a number.")
        categories.setdefault(str(category).strip(),
                              {"category": str(category).strip(),
                               "title_rule": {}, "image_slots": []})
        categories[str(category).strip()]["image_slots"].append({
            "slot": slot_num,
            "image_type": (image_type or "").strip(),
            "description": (description or "").strip(),
        })

    if TITLES_SHEET in wb.sheetnames:
        tws = wb[TITLES_SHEET]
        for row in tws.iter_rows(min_row=2, values_only=True):
            values = list(row) + [None] * (len(TITLE_FIELDS) + 1)
            category = (values[0] or "").strip() if values[0] else ""
            if category in categories:
                title_rule = {
                    key: (values[i + 1] or "").strip()
                    for i, (key, _) in enumerate(TITLE_FIELDS)
                }
                # A category with no title rules at all stays {}, so the
                # xlsx -> JSON -> xlsx round trip is byte-for-byte stable.
                categories[category]["title_rule"] = (
                    title_rule if any(title_rule.values()) else {}
                )

    for entry in categories.values():
        entry.setdefault("aliases", [])
    if ALIASES_SHEET in wb.sheetnames:
        aws = wb[ALIASES_SHEET]
        for row in aws.iter_rows(min_row=2, values_only=True):
            category = (row[0] or "").strip() if row and row[0] else ""
            alias = (row[1] or "").strip() if row and len(row) > 1 and row[1] else ""
            if category in categories and alias:
                categories[category]["aliases"].append(alias)

    rule_master = list(categories.values())
    for entry in rule_master:
        entry["image_slots"].sort(key=lambda s: s["slot"])
    return rule_master


# ---------------------------------------------------------------------
# Validation + writers
# ---------------------------------------------------------------------
def validate(rule_master: list) -> list:
    """Return a list of human-readable warnings. The pipeline assumes 6
    consecutively-numbered slots per category, so flag anything else rather
    than letting it surface as a confusing status three steps later."""
    warnings = []
    category_names = {entry["category"] for entry in rule_master}
    alias_owner = {}
    for entry in rule_master:
        slots = [s["slot"] for s in entry["image_slots"]]
        if sorted(slots) != list(range(1, 7)):
            warnings.append(f"{entry['category']}: expected slots 1-6, got {sorted(slots)}")
        if len(slots) != len(set(slots)):
            warnings.append(f"{entry['category']}: duplicate slot numbers {sorted(slots)}")
        for slot in entry["image_slots"]:
            if not slot["image_type"]:
                warnings.append(f"{entry['category']} slot {slot['slot']}: blank image type")
            if not slot["description"]:
                warnings.append(f"{entry['category']} slot {slot['slot']}: blank description")
        for alias in entry.get("aliases", []):
            if alias in category_names:
                warnings.append(f"{entry['category']}: alias {alias!r} is itself a real "
                                f"category name — this alias will never be reached")
            if alias in alias_owner and alias_owner[alias] != entry["category"]:
                warnings.append(f"alias {alias!r} is claimed by both "
                                f"{alias_owner[alias]!r} and {entry['category']!r}")
            alias_owner[alias] = entry["category"]
    return warnings


def write_json(rule_master: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rule_master, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(rule_master)} categories -> {path}")


FONT_NAME = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F2937")
HEADER_FONT = Font(name=FONT_NAME, size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name=FONT_NAME, size=10)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _write_sheet(ws, headers, widths, rows, row_height):
    ws.sheet_view.showGridLines = False
    for i, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=i, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = WRAP
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(i)].width = widths[i - 1]
    ws.freeze_panes = "A2"

    for r_idx, values in enumerate(rows, start=2):
        for c_idx, value in enumerate(values, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.font = BODY_FONT
            cell.alignment = WRAP
            cell.border = BORDER
        ws.row_dimensions[r_idx].height = row_height


def write_xlsx(rule_master: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = SLOTS_SHEET
    _write_sheet(
        ws,
        ["Category", "Slot", "Image Type", "Description"],
        [26, 8, 30, 70],
        [(entry["category"], slot["slot"], slot["image_type"], slot["description"])
         for entry in rule_master for slot in entry["image_slots"]],
        42,
    )

    # Kept in the workbook so the xlsx round-trips losslessly back to JSON.
    tws = wb.create_sheet(TITLES_SHEET)
    _write_sheet(
        tws,
        ["Category"] + [label for _, label in TITLE_FIELDS],
        [26, 40, 40, 34, 34, 34],
        [tuple([entry["category"]]
               + [entry.get("title_rule", {}).get(key, "") for key, _ in TITLE_FIELDS])
         for entry in rule_master],
        60,
    )

    # One row per alias (a category can have zero, one, or several), e.g.
    # "Hot Wheels" -> "Cars & RC Toys". Edit this sheet + re-run
    # --from xlsx to add more without touching code.
    aws = wb.create_sheet(ALIASES_SHEET)
    _write_sheet(
        aws,
        ["Category", "Alias (admin-panel name that should match this category)"],
        [26, 50],
        [(entry["category"], alias)
         for entry in rule_master for alias in entry.get("aliases", [])],
        20,
    )

    wb.save(path)
    print(f"Wrote {len(rule_master)} categories -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", required=True,
                    choices=["csv", "json", "xlsx"],
                    help="where to read the rules from: 'csv' = raw Cataloging 2.0 "
                         "export (needs --src), 'json' = the rules JSON, "
                         "'xlsx' = the human-editable sheet")
    ap.add_argument("--src", help="path to the raw CSV export (required for --from csv)")
    ap.add_argument("--json-out", default=DEFAULT_JSON)
    ap.add_argument("--xlsx-out", default=DEFAULT_XLSX)
    args = ap.parse_args()

    if args.source == "csv":
        if not args.src:
            ap.error("--from csv requires --src /path/to/the/export.csv")
        if not os.path.exists(args.src):
            sys.exit(f"ERROR: {args.src} not found.")
        rule_master = parse_source_csv(args.src)
        targets = ["json", "xlsx"]
    elif args.source == "json":
        if not os.path.exists(args.json_out):
            sys.exit(f"ERROR: {args.json_out} not found.")
        with open(args.json_out, encoding="utf-8") as f:
            rule_master = json.load(f)
        targets = ["xlsx"]
    else:
        if not os.path.exists(args.xlsx_out):
            sys.exit(f"ERROR: {args.xlsx_out} not found.")
        rule_master = parse_xlsx(args.xlsx_out)
        targets = ["json"]

    for warning in validate(rule_master):
        print(f"WARNING: {warning}")

    if "json" in targets:
        write_json(rule_master, args.json_out)
    if "xlsx" in targets:
        write_xlsx(rule_master, args.xlsx_out)


if __name__ == "__main__":
    main()
