"""
Shared helpers for the OZi Toys image pipeline.

Holds the things more than one pipeline step needs to agree on, so they
can't drift apart:
  1. Loading the rule master and matching a product's category to it
  2. The Covered_Slots / Missing_Slots CSV encoding
  3. Fetching image bytes (http(s) URL or local file, for demo mode)
  4. Uploading generated images to GCS and getting back a public link
  5. Checkpointing + bounded concurrency for bulk (1000+ product) runs
"""
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from urllib.parse import quote

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# Slot maps are stored in one CSV cell as "1:<value>; 2:<value>".
# Values (image URLs) contain ":" themselves, so parsing always splits on
# the FIRST colon only.
SLOT_MAP_SEPARATOR = "; "


def _normalize_category(name: str) -> str:
    """Fold the cosmetic differences between an admin-panel category name
    and the same category in the rule master: case, spacing, '&' vs 'and',
    and trailing punctuation. 'Art, Craft & DIYs' == 'art craft and diys'."""
    if not isinstance(name, str):
        return ""
    text = name.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


class RuleMaster:
    """The category -> 6 image slots rule set, with tolerant lookup.

    An entry may carry an optional "aliases" list — admin-panel category
    names that should resolve to it even though they're not its actual
    name, e.g. "Hot Wheels" (a brand, filed as its own sub-category in the
    admin panel) aliasing to "Cars & RC Toys". Aliases resolve to the
    canonical entry, so match() always returns the real category name.
    """

    def __init__(self, categories: list):
        self.categories = categories
        self._by_normalized = {}
        for entry in categories:
            self._by_normalized[_normalize_category(entry["category"])] = entry
            for alias in entry.get("aliases", []):
                self._by_normalized[_normalize_category(alias)] = entry

    @classmethod
    def load(cls, path: str) -> "RuleMaster":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path}: expected a non-empty list of categories")
        for entry in data:
            if "category" not in entry or "image_slots" not in entry:
                raise ValueError(f"{path}: entry missing 'category' or 'image_slots': {entry!r}")
        return cls(data)

    def match(self, *category_names) -> tuple:
        """Resolve a product's category against the rule master, trying each
        name in turn (normally Category_L1, then L2, then L3 — products are
        sometimes filed one level deeper than the rules are written).

        Returns (matched_category_name, {slot_num: slot_dict}) or ("", {}).
        """
        for name in category_names:
            entry = self._by_normalized.get(_normalize_category(name))
            if entry:
                slots = {s["slot"]: s for s in entry["image_slots"]}
                return entry["category"], slots
        return "", {}

    @property
    def category_names(self) -> list:
        return [c["category"] for c in self.categories]


def format_slot_map(slot_to_value: dict) -> str:
    """{1: 'https://…'} -> '1:https://…'"""
    return SLOT_MAP_SEPARATOR.join(
        f"{slot}:{slot_to_value[slot]}" for slot in sorted(slot_to_value)
    )


def parse_slot_map(text: str) -> dict:
    """'1:https://…; 2:https://…' -> {1: 'https://…', 2: 'https://…'}"""
    result = {}
    if not isinstance(text, str):
        return result
    for part in text.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        slot, value = part.split(":", 1)
        try:
            result[int(slot.strip())] = value.strip()
        except ValueError:
            continue
    return result


def split_image_urls(text: str) -> list:
    if not isinstance(text, str):
        return []
    return [u.strip() for u in text.split(";") if u.strip()]


def get_image_bytes(path_or_url: str, timeout: int = 20) -> tuple:
    """Read an image from an http(s) URL (real pipeline) or a local path
    (demo mode). Returns (bytes, media_type)."""
    if path_or_url.startswith(("http://", "https://")):
        resp = requests.get(path_or_url, timeout=timeout)
        resp.raise_for_status()
        media_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not media_type.startswith("image/"):
            media_type = _media_type_from_extension(path_or_url)
        return resp.content, media_type
    with open(path_or_url, "rb") as f:
        return f.read(), _media_type_from_extension(path_or_url)


def _media_type_from_extension(path: str) -> str:
    ext = os.path.splitext(path.split("?", 1)[0])[1].lower().lstrip(".")
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "webp": "image/webp", "gif": "image/gif",
    }.get(ext, "image/jpeg")


def parse_description_fields(description: str) -> dict:
    """OZi descriptions are a free-text blurb followed by structured
    "Key: Value" lines (one per line), e.g.:
        Gender: Unisex
        Dimensions / Size: 10 x 10 x 3 cm
        Battery Operated (Yes/No): No
    Returns {lowercased key: value}, or {} if description is blank/not a
    structured field list."""
    if not isinstance(description, str) or not description:
        return {}
    fields = {}
    for line in description.replace("\r\n", "\n").split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()
    return fields


def merge_product_fields(description: str, specifications: str = "") -> dict:
    """The admin panel has TWO independent sources of "Key: Value" facts
    about a product:
      - the free-text Description's structured tail (parsed by
        parse_description_fields) — sometimes stale, sometimes uses a
        different key for the same idea (e.g. a bare "Size" describing an
        action figure's own scale)
      - the dedicated "Specification" section (Brand, "Dimensions (LxBxH)",
        Weight, Battery Operated, ...) — a proper structured dict in the
        admin panel, not just embedded prose

    These can genuinely disagree (a 4-inch Iron Man figure inside a
    25 x 24 x 7 cm display-chamber box — both true, different facts) and
    the Specification section is the more reliable, deliberately-entered
    one for anything about the physical product/box, so it wins on any key
    collision. Callers should read from this merged dict, not either field
    directly.
    """
    merged = parse_description_fields(description)
    merged.update(parse_description_fields(specifications))
    return merged


_AXIS_LABELS = {"l": "Length", "w": "Width", "b": "Breadth", "h": "Height", "d": "Depth"}


def _label_dimension_value(key: str, value: str) -> str:
    """Given a key like "dimensions (lxbxh)" and a value like
    "26.2 x 14.1 x 4.8 cm", return "Length 26.2 cm, Breadth 14.1 cm,
    Height 4.8 cm" so each number is tied to a real axis instead of being
    a bare, order-dependent triple the model has to guess at.

    A real product came back with the model drawing only 2 arrows and
    mislabeling the vertical one 14.1 cm (the Breadth) instead of 4.8 cm
    (the actual Height) — it had no way to know which of the 3 raw numbers
    was which axis. Returns the original value unchanged if the key
    doesn't carry an explicit axis-order hint (e.g. plain "Dimensions /
    Size"), so callers with no letter hint keep today's behavior.

    "Length" is always re-paired to whichever of the two HORIZONTAL
    numbers (Length vs Breadth/Width) is bigger — Height/Depth is never
    touched. A real product's own Specifications literally read "Dimensions
    (LxBxH): 22.8 x 28.5 x 3.8 cm" (L=22.8 < B=28.5) — trusting that literal
    pairing rendered a correct, self-consistent image (the bigger 28.5 edge
    genuinely drawn longer), but a reviewer expects the edge LABELED
    "Length" to always be the longer one, regardless of which word the
    admin happened to write next to which number. This is a deliberate
    product decision (confirmed explicitly), not a bug fix — some real
    admin data pairs these two words with the smaller/bigger number either
    way.
    """
    axis_match = re.search(r"\b([lwbhd])x([lwbhd])x([lwbhd])\b", key)
    if not axis_match:
        return value
    numbers = re.findall(r"[\d.]+", value)
    if len(numbers) != 3:
        return value
    unit_match = re.search(r"([a-zA-Z]+)\s*$", value.strip())
    unit = unit_match.group(1) if unit_match else ""
    letters = list(axis_match.groups())
    length_idx = [i for i, l in enumerate(letters) if l == "l"]
    other_horiz_idx = [i for i, l in enumerate(letters) if l in ("w", "b")]
    if len(length_idx) == 1 and len(other_horiz_idx) == 1:
        li, oi = length_idx[0], other_horiz_idx[0]
        if float(numbers[li]) < float(numbers[oi]):
            numbers[li], numbers[oi] = numbers[oi], numbers[li]
    labeled = [f"{_AXIS_LABELS[letter]} {num} {unit}".strip()
              for letter, num in zip(letters, numbers)]
    return ", ".join(labeled)


def extract_dimensions_from_description(description: str, specifications: str = "") -> str:
    """Pull the real manufacturer-listed size out of the admin panel data,
    so the "Size / Dimensions" generated image can draw the actual numbers
    instead of the model inventing plausible-looking but wrong ones.

    Looks for a key containing "dimension" (covers "Dimensions",
    "Dimensions (LxBxH)", "Dimensions / Size", etc.), falling back to an
    exact "Size" key (not "Pack Size" — checked by exact match, not
    substring, precisely to avoid that false positive). Returns "" if
    neither source has one — real data shows this happens for a minority
    of products, and the caller falls back to letting the model estimate
    proportionately in that case.

    A real product had the SAME dimensions under two differently-spelled
    keys — Description's plain "Dimensions / Size" (no axis-order hint)
    and Specifications' "Dimensions (LxBxH)" (has one) — so merging into
    one dict (merge_product_fields) didn't help: the two keys don't
    collide, both survive, and picking "whichever comes first" could land
    on the unlabeled one even though the labeled one is right there. This
    checks Specifications and Description as separate sources (Specification
    wins outright when it has any dimension key at all, matching its
    documented authority), and within a source, prefers whichever key
    carries an axis-order hint (see _label_dimension_value) over one that
    doesn't, rather than just the first key found.
    """
    spec_fields = parse_description_fields(specifications)
    desc_fields = parse_description_fields(description)
    axis_hint_re = re.compile(r"\b([lwbhd])x([lwbhd])x([lwbhd])\b")
    for fields in (spec_fields, desc_fields):
        candidates = [(k, v) for k, v in fields.items() if "dimension" in k and v]
        if not candidates:
            continue
        hinted = [(k, v) for k, v in candidates if axis_hint_re.search(k)]
        key, value = hinted[0] if hinted else candidates[0]
        return _label_dimension_value(key, value)
    return spec_fields.get("size") or desc_fields.get("size", "")


DIMENSION_STATUS_VERIFIED = "VERIFIED_DIMENSIONS"
DIMENSION_STATUS_AMBIGUOUS = "AMBIGUOUS_DIMENSIONS"
DIMENSION_STATUS_MISSING = "MISSING_DIMENSIONS"

_AXIS_VALUE_RE = re.compile(r"(Length|Width|Breadth|Height|Depth)\s+([\d.]+)\s*([a-zA-Z]*)")


def _parse_labeled_axis_values(labeled_text: str) -> dict:
    """"Length 26.2 cm, Breadth 14.1 cm, Height 4.8 cm" -> {"Length": 26.2,
    "Breadth": 14.1, "Height": 4.8, "_unit": "cm"}. {} if nothing parses —
    the caller treats that as "couldn't actually verify an axis" rather
    than trusting the raw text blindly."""
    out = {}
    unit = ""
    for axis, num, u in _AXIS_VALUE_RE.findall(labeled_text):
        out[axis] = float(num)
        unit = unit or u
    if out:
        out["_unit"] = unit
    return out


def classify_dimensions(description: str, specifications: str = "") -> dict:
    """The single source of truth for whether a product's admin-panel size
    data can be trusted to say WHICH number is Length vs Breadth vs Height
    — not just that three numbers exist somewhere in the text.

    extract_dimensions_from_description() already picks the right field and,
    when the key carries an explicit axis-order hint (e.g. "Dimensions
    (LxBxH)"), labels each number by name via _label_dimension_value(). But
    it returns a plain string either way — callers previously treated "we
    found some numbers" as good enough to attempt arrow generation, even
    for a bare "Dimensions / Size: 20 x 15 x 2 cm" with NO order guarantee
    at all. That silently asked the image model to guess which of the two
    horizontal numbers was Length vs Breadth — exactly the axis-swap defect
    this function exists to stop before generation ever starts.

    Returns a dict:
        length, breadth, height  — float or None
        unit                     — str
        source                   — "Specifications" | "Description" | ""
        status                   — one of the three DIMENSION_STATUS_* constants
        raw_text                 — the original text this was derived from
        confidence               — "high" | "low" | "none"

    Rules:
      VERIFIED_DIMENSIONS — the source key carries an axis-order hint
        (LxBxH-style) AND the labeled numbers actually parsed. Safe to
        build a Measurement Layout Plan and check drawn arrows against it.
      AMBIGUOUS_DIMENSIONS — real numbers exist (a dimension key, or a bare
        "Size" field) but nothing proves the order is Length-then-Breadth-
        then-Height. Do NOT let the image model guess this order — route
        to manual review instead (see generate_missing_images_gcp.py).
      MISSING_DIMENSIONS — no usable size data at all.
    """
    spec_fields = parse_description_fields(specifications)
    desc_fields = parse_description_fields(description)
    axis_hint_re = re.compile(r"\b([lwbhd])x([lwbhd])x([lwbhd])\b")

    for source_name, fields in (("Specifications", spec_fields), ("Description", desc_fields)):
        candidates = [(k, v) for k, v in fields.items() if "dimension" in k and v]
        if not candidates:
            continue
        hinted = [(k, v) for k, v in candidates if axis_hint_re.search(k)]
        if hinted:
            key, value = hinted[0]
            labeled = _label_dimension_value(key, value)
            axis_values = _parse_labeled_axis_values(labeled)
            if axis_values:
                return {
                    "length": axis_values.get("Length"),
                    "breadth": axis_values.get("Breadth", axis_values.get("Width")),
                    "height": axis_values.get("Height"),
                    "unit": axis_values.get("_unit", ""),
                    "source": source_name,
                    "status": DIMENSION_STATUS_VERIFIED,
                    "raw_text": labeled,
                    "confidence": "high",
                }
        # A dimension key exists but with no axis-order hint (or the hinted
        # key's numbers didn't parse cleanly) — real numbers, unproven
        # order. AMBIGUOUS, not VERIFIED, even though
        # extract_dimensions_from_description() would return them as-is.
        key, value = candidates[0]
        return {
            "length": None, "breadth": None, "height": None, "unit": "",
            "source": source_name, "status": DIMENSION_STATUS_AMBIGUOUS,
            "raw_text": value, "confidence": "low",
        }

    bare_size = spec_fields.get("size") or desc_fields.get("size", "")
    if bare_size:
        return {
            "length": None, "breadth": None, "height": None, "unit": "",
            "source": "Specifications" if spec_fields.get("size") else "Description",
            "status": DIMENSION_STATUS_AMBIGUOUS, "raw_text": bare_size,
            "confidence": "low",
        }
    return {
        "length": None, "breadth": None, "height": None, "unit": "",
        "source": "", "status": DIMENSION_STATUS_MISSING, "raw_text": "",
        "confidence": "none",
    }


def is_battery_operated(description: str, specifications: str = ""):
    """True/False from a "Battery Operated (Yes/No): ..." field (either
    source — see merge_product_fields), or None if neither says. Used to
    stop the model inventing a controller/remote/batteries for products
    that don't have any — the generic per-category slot instructions
    (written to cover a whole category like "Cars & RC Toys") mention a
    "controller" even for plain die-cast items with none, and the model
    was taking that literally."""
    fields = merge_product_fields(description, specifications)
    for key, value in fields.items():
        if "battery" in key and value:
            return value.strip().lower().startswith("y")
    return None


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).strip()).strip("_").lower()


def write_link_cell(ws, row: int, col: int, value, url: str = None, font=None, border=None):
    """Write a cell that's an actual clickable hyperlink when `url` is a real
    link, not just text that happens to look like one. A plain string
    starting with "http" is NOT a link to Excel/Sheets unless the cell's
    `.hyperlink` attribute is set — this is why URLs written as plain
    values don't open when clicked."""
    from openpyxl.styles import Alignment, Font

    cell = ws.cell(row=row, column=col, value=value)
    if url and url.startswith(("http://", "https://")):
        cell.hyperlink = url
        cell.font = Font(name=font.name if font else "Arial",
                         size=font.size if font else 10,
                         color="0563C1", underline="single")
    elif font:
        cell.font = font
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    if border:
        cell.border = border
    return cell


class VertexTokenProvider:
    """Supplies a live Vertex AI access token, refreshed on demand.

    Service-account tokens expire after ~1 hour. A 2000-product bulk run
    makes thousands of calls over many hours, so a token fetched once at
    startup would expire partway through and 401 every remaining request —
    every caller should ask this for a fresh token rather than caching one
    itself.
    """

    def __init__(self, key_path: str):
        if not key_path:
            sys.exit("ERROR: set GOOGLE_APPLICATION_CREDENTIALS to your "
                     "service account JSON file path.")
        if not os.path.exists(key_path):
            sys.exit(f"ERROR: GOOGLE_APPLICATION_CREDENTIALS points at "
                     f"{key_path!r}, which does not exist.")
        try:
            self._creds = service_account.Credentials.from_service_account_file(
                key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except (ValueError, KeyError) as e:
            sys.exit(f"ERROR: {key_path!r} is not a valid service account key file: {e}")
        # Bulk runs call this from many worker threads at once (see
        # concurrency in classify_images.py / generate_missing_images_gcp.py).
        # Without a lock, two threads finding an expired token at the same
        # moment both call _creds.refresh() concurrently, which mutates the
        # same Credentials object's internal token/expiry state from two
        # threads at once — a real race, not a hypothetical one now that
        # multiple threads exist. The lock only serializes the brief
        # check-and-maybe-refresh; reading an already-valid token is cheap.
        self._lock = threading.Lock()

    def token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if force_refresh or not self._creds.valid:
                self._creds.refresh(Request())
            return self._creds.token


class Checkpoint:
    """Append-only JSONL progress log so a crashed or interrupted bulk run
    can resume instead of redoing already-completed work — both to avoid
    re-paying for API calls that already succeeded, and so a multi-hour
    2000-product run surviving a restart is actually practical.

    Usage: give every unit of work a stable key (e.g. "<product_id>:<slot>"
    or just "<product_id>" for a whole-row step). Before doing the work,
    check is_done(key) and reuse get(key) if so. After it succeeds, call
    record(key, row) — this appends to disk and fsyncs immediately, so a
    kill -9 right after only ever loses the one in-flight unit of work,
    never anything already recorded.

    Thread-safe: record() is called from worker threads under concurrency
    (see run_concurrent below), so writes are serialized with a lock —
    each line is a single atomic write, and json.dumps + "\\n" never
    interleaves with another thread's line.
    """

    def __init__(self, path: str):
        self.path = path
        self._done = {}
        self._lock = threading.Lock()
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._done[entry["key"]] = entry["row"]
        except FileNotFoundError:
            pass
        self._fh = open(path, "a", encoding="utf-8")

    def is_done(self, key: str) -> bool:
        return key in self._done

    def get(self, key: str):
        return self._done.get(key)

    def record(self, key: str, row) -> None:
        with self._lock:
            self._done[key] = row
            line = json.dumps({"key": key, "row": row})
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def run_concurrent(tasks: list, worker_fn, max_workers: int, on_result=None) -> list:
    """Runs worker_fn(task) over a bounded thread pool (I/O-bound HTTP
    calls, so threads — not processes — are the right tool) and returns
    results in the SAME ORDER as tasks, not completion order, so callers
    can zip results back up with their inputs positionally.

    max_workers doubles as the simple, effective rate-limit control here:
    Vertex AI's per-minute quota is a concurrent/rate limit, and bounding
    how many requests are ever in flight at once (rather than trying to
    precisely pace requests/minute) is enough given every real caller here
    already retries 429/500 with backoff on top of this.

    worker_fn is expected to catch its own errors and return an
    error-shaped result (every API-calling function in this pipeline
    already does — e.g. {"status": "failed: ..."} rather than raising) —
    this does not swallow exceptions, so a worker_fn that raises will
    still surface via future.result() and stop the run, same as any
    uncaught exception would.

    on_result(index, task, result), if given, fires as each result comes
    in (e.g. to checkpoint it immediately or print progress) — called from
    whichever worker thread finished, so anything it touches must be its
    own thread-safe (Checkpoint.record already is).
    """
    results = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(worker_fn, task): i
                           for i, task in enumerate(tasks)}
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            result = future.result()
            results[i] = result
            if on_result:
                on_result(i, tasks[i], result)
    return results


def write_status(path: str, **fields) -> None:
    """Best-effort progress snapshot for basic monitoring on an unattended
    VM run — a 1-2k product run has no human watching a terminal, so
    something needs to be pollable from outside (a cron job, a health
    check, or just `cat <out>.status.json` over SSH) to notice a stall
    without tailing logs. Written atomically (temp file + rename) so a
    poller never reads a half-written file; every call overwrites the
    whole snapshot, so this is meant to be called periodically (e.g. from
    an on_result callback every N completions), not on every single one.
    """
    fields["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2)
    os.replace(tmp_path, path)


class GcsUploadError(Exception):
    """Raised for a permanent (non-retryable) upload failure — bad bucket
    name, credentials rejected, etc. A caller should stop, not skip a row
    and keep going, since it will fail identically for every remaining row."""


class GcsUploader:
    """Uploads generated images to a GCS bucket and returns a public URL.

    Public access works two different ways depending on the bucket's
    settings, and a service account key alone can't tell you which one
    applies:
      - Fine-grained ACLs (uniform bucket-level access OFF): the upload
        itself can request `predefinedAcl=publicRead` and the object is
        public immediately.
      - Uniform bucket-level access ON (Google's current default for new
        buckets): per-object ACLs are rejected outright; the object is
        public only if the BUCKET has an IAM binding of
        allUsers -> Storage Object Viewer. This class can't grant that —
        it just detects the rejection and falls back to plain upload,
        with a one-time warning so a link doesn't quietly point at a
        private object.
    """

    UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
    PATCH_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    MAX_RETRIES = 3
    RETRYABLE_STATUS = {401, 408, 429, 500, 502, 503, 504}

    def __init__(self, key_path: str, bucket: str, prefix: str = "", public: bool = True):
        if not os.path.exists(key_path):
            sys.exit(f"ERROR: GCS credentials file not found: {key_path!r}")
        try:
            self._creds = service_account.Credentials.from_service_account_file(
                key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except (ValueError, KeyError) as e:
            sys.exit(f"ERROR: {key_path!r} is not a valid service account key file: {e}")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.public = public
        self._uniform_bucket_access = False  # flips true after the first ACL rejection
        self._warned_about_uniform_access = False

    def _token(self, force_refresh: bool = False) -> str:
        if force_refresh or not self._creds.valid:
            self._creds.refresh(Request())
        return self._creds.token

    def object_name(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def _set_no_cache(self, object_name: str):
        """GCS's default Cache-Control (public, max-age=3600) means a
        product image regenerated under the same filename can serve the
        OLD bytes to a browser for up to an hour after the new upload —
        this bit us during testing (a corrected image looked unchanged
        until a hard refresh). Best-effort: a failure here doesn't fail
        the upload, since the image itself is already live either way."""
        try:
            requests.patch(
                f"{self.PATCH_URL.format(bucket=self.bucket)}/{quote(object_name, safe='')}",
                headers={"Authorization": f"Bearer {self._token()}",
                        "Content-Type": "application/json"},
                json={"cacheControl": "no-cache, max-age=0, must-revalidate"},
                timeout=15,
            )
        except requests.RequestException:
            pass

    def public_url(self, filename: str) -> str:
        return f"https://storage.googleapis.com/{self.bucket}/{quote(self.object_name(filename))}"

    def upload(self, data: bytes, filename: str, content_type: str) -> str:
        """Upload bytes to {bucket}/{prefix}/{filename}. Returns the public
        URL on success, raises GcsUploadError on a permanent failure."""
        object_name = self.object_name(filename)
        params = {"uploadType": "media", "name": object_name}
        if self.public and not self._uniform_bucket_access:
            params["predefinedAcl"] = "publicRead"

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            headers = {"Authorization": f"Bearer {self._token()}", "Content-Type": content_type}
            try:
                resp = requests.post(
                    self.UPLOAD_URL.format(bucket=self.bucket),
                    headers=headers, params=params, data=data, timeout=60,
                )
            except requests.RequestException as e:
                last_error = str(e)
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 * attempt)
                continue

            if resp.status_code == 200:
                self._set_no_cache(object_name)
                return self.public_url(filename)

            # Bucket has uniform bucket-level access on — predefinedAcl is
            # rejected outright. Retry the SAME attempt without it instead
            # of burning a retry slot, and remember for every later call.
            if (resp.status_code == 400 and "predefinedAcl" in params
                    and ("uniform bucket-level access" in resp.text.lower()
                         or "predefinedacl" in resp.text.lower())):
                self._uniform_bucket_access = True
                if not self._warned_about_uniform_access:
                    print("WARNING: bucket has uniform bucket-level access enabled — "
                          "per-object public ACLs are rejected. Uploading without one; "
                          "the object will only be publicly reachable at the URL in "
                          "GCP_Link if the BUCKET itself grants allUsers the "
                          "'Storage Object Viewer' role. Check Bucket -> Permissions "
                          "in the GCP Console if links come back 403.")
                    self._warned_about_uniform_access = True
                params.pop("predefinedAcl")
                continue  # same attempt count; retry immediately, not a backoff case

            if resp.status_code == 401:
                self._token(force_refresh=True)

            if resp.status_code not in self.RETRYABLE_STATUS:
                raise GcsUploadError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if attempt < self.MAX_RETRIES:
                time.sleep(2 * attempt)

        raise GcsUploadError(f"upload failed after {self.MAX_RETRIES} attempts: {last_error}")
