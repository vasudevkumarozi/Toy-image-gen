"""
Step 3 (Google/Vertex AI version): Generate Missing Images + Final Output Sheet
==================================================================================
Uses Vertex AI's Gemini Image model ("Nano Banana") for image EDITING —
reference product photo + text prompt in, edited image out — so the toy
itself is preserved and only the scene/angle changes.

Note: Google's older Imagen edit API (imagen-3.0-capability-001) was
deprecated in 2026. Gemini Image models are now the supported path for
this on Vertex AI.

Reads:
  - products_detail.csv        (from fetch_product_details.py)
  - classification_result.csv  (from classify_images.py)
  - toys_rule_master.json      (category -> slot descriptions)

For every MISSING slot: picks a reference image (prefers slot 1/front,
else any covered slot, else the first raw image), calls Vertex AI Gemini
Image with that reference + a rule-based prompt, and saves the result to
generated_images/{SKU}_{slot}_{type}.png

Already-generated files are reused instead of re-billed; pass --overwrite
to regenerate them.

Writes final_output.xlsx (long format, one row per product per slot):
  Product_ID, SKU, Name, Category, Slot, Image_Type, Status,
  Image_Source, GCP_Link

GCP_Link is filled in automatically when GCS upload is configured (see
GCS_BUCKET below); otherwise it's left blank for you to fill in by hand
after manually uploading the files in --image_out_dir.

SECURITY / CREDENTIALS:
  - GOOGLE_APPLICATION_CREDENTIALS -> path to your GCP service account
    JSON key file, for Vertex AI image generation (never the key content
    itself, just the file path)
  - GCP_PROJECT_ID                 -> your GCP project id
  - GCP_REGION                     -> defaults to "us-central1"
    (Note: the higher-quality "Pro" model may require the "global"
    Vertex AI endpoint rather than a regional one — check your project's
    Model Garden access if you hit a 404/NOT_FOUND on the region below.)

  Optional — auto-upload generated images to GCS and fill in GCP_Link:
  - GCS_BUCKET      -> bucket name. May include a "/prefix" suffix, e.g.
                        "my-bucket/toys/generated" — everything after the
                        first "/" is used as the object path prefix.
  - GCS_PREFIX       -> object path prefix within the bucket (optional;
                        combined with any prefix already in GCS_BUCKET)
  - GCS_CREDENTIALS  -> path to the service account JSON key with Storage
                        write access. Can be the SAME file as
                        GOOGLE_APPLICATION_CREDENTIALS if that account also
                        has a Storage role, or a different one.
  Uploaded objects are made publicly readable via predefinedAcl=publicRead;
  if the bucket has uniform bucket-level access enabled, that's rejected
  and the script falls back to a plain upload with a one-time warning —
  make the BUCKET itself public (allUsers -> Storage Object Viewer) in
  that case, or the GCP_Link URLs won't be reachable.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
    export GCP_PROJECT_ID="your-project-id"
    python3 generate_missing_images_gcp.py \
        --products products_detail.csv \
        --classification classification_result.csv \
        --rules toys_rule_master.json \
        --image_out_dir generated_images \
        --out final_output.xlsx
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time

import openpyxl
import pandas as pd
import requests
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from PIL import Image
from io import BytesIO

from pipeline_lib import (
    Checkpoint,
    GcsUploader,
    GcsUploadError,
    RuleMaster,
    VertexTokenProvider,
    extract_dimensions_from_description,
    is_battery_operated,
    get_image_bytes,
    parse_slot_map,
    run_concurrent,
    slugify,
    split_image_urls,
    write_link_cell,
    write_status,
)

# "gemini-3.1-flash-image" ("Nano Banana 2") is NOT enabled in the
# `ozitech` GCP project — confirmed via a live 404 ("Publisher model ...
# was not found") both from the CLI and, once more, from a real UI run
# that had no manual override set. Every CLI invocation all session long
# worked around this with an ad-hoc `export GEMINI_IMAGE_MODEL=...`
# that was never written back here or into .env, so anything that didn't
# know about that undocumented step (like the UI, or anyone else running
# this fresh) hit the same 404. gemini-2.5-flash-image is the model
# actually verified working in this project — that has to be the
# default, not an override you have to remember.
MODEL_ID = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
MAX_RETRIES = 3
# Worth another attempt: throttling, transient backend faults, and a stale
# token (retried once after a forced refresh). Everything else — 400 bad
# request, 403 missing role, 404 unknown model — fails the same way forever.
RETRYABLE_STATUS = {401, 408, 429, 500, 502, 503, 504}

# Native output size, guaranteed by upscaling below regardless of what the
# model actually returns. gemini-2.5-flash-image (the only image model
# enabled on this project as of testing) ignores the imageConfig.imageSize
# hint entirely and always returns 1024x1024 — verified empirically across
# "1K"/"2K"/"4K" with no change in output pixel size. We still send the
# hint (harmless) in case a future/enabled model honors it, and enforce the
# floor ourselves so the guarantee doesn't silently depend on that.
MIN_OUTPUT_PX = 2048

LOCK_RULES = (
    "Do not change the product's color, shape, logo, material, texture, "
    "proportions or accessories. The product must remain exactly as shown "
    "in the reference image — only the scene, background, or angle changes "
    "as instructed below."
)

# Deliberately does NOT say "show the entire product" — many slot types
# (e.g. "Feature": "focus on one distinct and relevant part") are supposed
# to be a close-up crop on purpose. An earlier version of this rule said
# "keep the ENTIRE product visible", which directly contradicted those
# close-up requirements — the model resolved the conflict by drawing two
# overlapping views (a wide shot AND a close-up stacked together), which
# was worse than the original problem. This version only forbids the shot
# looking ACCIDENTALLY clipped, whatever the intended subject is.
# The softer wording of this rule still let real crops through — a
# multi-panel product (e.g. a busy book) came back with several panels
# sliced off at the edges because the model tried to fit everything at
# too tight a zoom instead of pulling back. Spelled out explicitly what
# "don't crop" means operationally: zoom OUT, don't zoom to fit.
FRAMING_RULE = (
    "Whatever this shot's subject is meant to be — the whole product or a "
    "close-up on one detail — every part of THAT subject must be fully "
    "inside the frame with visible empty space on all four sides between "
    "it and the edge. If the intended subject doesn't fit with that margin "
    "at the current zoom, zoom OUT further — do not let any part of it "
    "(a panel, a corner, an edge, a limb) touch or extend past the image "
    "border. This applies even to multi-panel or multi-part products: show "
    "less of the product rather than cropping part of what you do show. "
    "This ALSO applies to every distinct included accessory shown alongside "
    "the main item (e.g. a baby-doll accessory held by a larger doll) — a "
    "secondary component being cut off at the edge is just as much a crop "
    "error as the main product being cut off. "
    "If the product is a figure with a head and limbs (a doll, action "
    "figure, plush animal) or has a continuous body/silhouette (a "
    "vehicle) rather than separate independent parts: a close-up must not "
    "cut off the head, face, hands, or feet at the frame edge — moving the "
    "crop to exclude one problem (e.g. a second figure) but then cutting "
    "off the SAME figure's own head or hands instead is not a fix, it's "
    "the identical error moved to a different part of the body. For a "
    "small humanoid doll or action figure that IS the product being "
    "catalogued: the doll itself must be the visible subject of this "
    "photo — do not solve a cropping problem by swapping the doll out for "
    "a close-up of only an accessory or background prop instead; the doll "
    "missing from the shot entirely is not an acceptable fix. Show the "
    "COMPLETE figure, head to feet, fully in frame, from a genuinely "
    "different pose or angle than earlier images — on a figure this "
    "small, a partial crop of just the torso is too close to both the "
    "head and the hands to reliably avoid cutting one of them off, so "
    "default to the whole figure rather than a partial one. If MORE THAN "
    "ONE figure appears together (e.g. a doll holding a baby-doll "
    "accessory), zoom out enough that EVERY figure's head, hands, and "
    "feet are fully inside the frame — do not pick a zoom level that fits "
    "one figure completely while cropping another; fit the widest of the "
    "two needs, not the tightest. If fitting a second figure completely "
    "is not possible at a close-up zoom, there is no partial option — "
    "either zoom out until it fits completely (head, hands, and feet all "
    "inside the frame), or reframe/recompose so NONE of that second "
    "figure is visible at all, not even a hand or a sliver at the edge. "
    "A second figure barely poking into the frame is the same crop error "
    "as one that is mostly cut off. "
    "Concrete check: this shot's zoom level must be no TIGHTER than the "
    "reference image's own framing — if you are unsure whether something "
    "fits, use a WIDER framing than the reference, never a tighter one. "
    "Show exactly one view; do not add a second overlapping or duplicate "
    "view of the product in the same image."
)

# A generated "rear/back angle" shot came back showing the doll's hair
# from behind AS IF still visible through a sealed blister-pack window,
# merged with printed box-back text/photos in the same frame — physically
# incoherent, since a sealed package can't show a rotated product. Forces
# a choice between the two coherent options instead.
PACKAGING_LOGIC_RULE = (
    "If this shot is meant to show a rear/back/opposite-side view and the "
    "product is normally sold in a box or blister pack: pick exactly ONE "
    "of these two — (a) the product fully removed from any packaging and "
    "rotated to show that view, with no box in the shot at all, or (b) the "
    "sealed retail package's own back panel (the printed cardboard side, "
    "with no product visible through any window). Never mix the two — the "
    "product cannot be both still sealed in a package AND visibly rotated "
    "to a different angle at the same time; that is not physically possible "
    "and reads as a broken image."
)

FOCUS_RULE = (
    "Keep the product AND its background in sharp, even focus throughout "
    "the whole image — no blur, no bokeh, no shallow depth of field."
)

# Telling the model "don't crop" wasn't sufficient by itself for a
# many-part product — it kept composing a scattered flat-lay of many
# separate pieces (a multi-panel busy book came back with several panels
# and crayons individually sliced off at the edges). Restricting how many
# things the shot tries to include removes the actual cause of that
# clutter, rather than asking the model to fit the same clutter more
# carefully.
FEATURE_STYLE_RULE = (
    "For a product with many small parts or panels, do NOT lay out a "
    "scattered collage of many separate pieces — that is what causes "
    "individual pieces to get cut off at the edges. Instead, pick just "
    "ONE or TWO of those parts/panels to feature, photographed like a "
    "hero product shot with generous empty background space around them, "
    "not packed edge-to-edge. "
    "For a product that is one continuous body or silhouette (e.g. a "
    "vehicle) rather than separate parts: do not frame a close-up that "
    "cuts the body off mid-length at the image edge (e.g. showing the "
    "front half of a car with the rear trailing off-frame) — that reads "
    "as a cropping error, not an intentional close-up. Instead, either "
    "(a) zoom in on one small, self-contained area only — like a single "
    "wheel and the small section of body immediately around it — with "
    "clear background visible beyond that area on every side and no other "
    "part of the body reaching the frame edge, or (b) show the complete "
    "item end to end. For option (a) on a vehicle specifically: shoot from "
    "a 3/4 angle (not a flat side profile) so the rest of the car recedes "
    "diagonally away from camera into soft background blur, rather than "
    "running parallel to the frame and getting sliced by a straight edge."
)

# Requested so a customer can tell what a "Feature" shot is actually
# demonstrating at a glance, without reading the full listing. The caption
# must name a REAL feature pulled from this specific product's own
# description/specification text — never an invented or generic one, same
# discipline as NO_INVENTED_ACCESSORIES_RULE below. Upgraded from a plain
# bottom caption band to match the polished callout style of established
# marketplace listing images (a reference example was provided): a
# magnified detail bubble or a pointer line/arrow connecting a short label
# straight to the specific part it names, not just text floating nearby.
FEATURE_TEXT_LABEL_RULE = (
    "This image must include a clear, visually appealing text label "
    "identifying the specific feature being shown, connected to that "
    "feature with a subtle pointer line or arrow so it's immediately "
    "obvious which part of the product the label refers to — do not just "
    "place text somewhere on the image with no visual connection to what "
    "it names. Use one of these professional ecommerce-listing callout "
    "styles: (a) a thin leader line running from a small text label (a "
    "rounded pill or tag background) directly to the exact spot on the "
    "product it describes, or (b) a circular or rounded-rectangle "
    "magnified inset bubble showing a zoomed-in close-up of that one "
    "detail, linked by a short connecting line to its label. Position the "
    "label/bubble in open background space, never on top of the product "
    "or overlapping another label. The label text must name ONE real, "
    "specific feature or selling point of THIS exact product, taken only "
    "from the product facts given above (its description bullets or "
    "specification fields) — do not invent a feature that isn't "
    "mentioned there. The pointer must land on the part that ACTUALLY "
    "performs that named function, not just a plausible-looking or "
    "conveniently nearby spot — a real image labeled \"Free Wheel "
    "Mechanism\" pointed at the mixer drum instead of the wheels/axle, "
    "which is wrong no matter how correct the label text itself is. "
    "Before placing each pointer, identify which specific visible part of "
    "the product is physically responsible for that feature, and point "
    "there specifically. Keep the label short (a few words), clearly "
    "legible, and set in a clean bold sans-serif font with good contrast "
    "against its background — polished and modern, like a real "
    "marketplace product-listing infographic, not handwritten."
)

# OZi's catalog is for the Indian market — Lifestyle shots were defaulting
# to generic/Western-looking models and settings, which doesn't represent
# the actual customer.
INDIA_REPRESENTATION_RULE = (
    "The people shown (children, parents, family members) must look Indian "
    "— Indian skin tones, facial features, and hair — not Western/European "
    "in appearance. The home or setting should read as a realistic Indian "
    "household/room (nothing needs to be stereotyped or overly ornate — an "
    "ordinary, realistic Indian home is right)."
)

# A real lifestyle shot for a single die-cast truck (sold as "1 Model Mixer
# Toy", nothing else) came back with a second, clearly-different toy
# vehicle (an excavator) placed right next to it in the scene — a customer
# glancing at that photo could reasonably think the excavator is included.
# Generic, non-branded play props (blocks, cones, a rug) don't have this
# problem since they don't look like a specific product being sold.
NO_OTHER_PRODUCTS_RULE = (
    "Do not add any other toy that looks like a distinct, separately "
    "sellable product into this scene — no second vehicle, figure, playset, "
    "or branded item that isn't THIS product. A customer must not be able "
    "to look at this photo and think something extra comes in the box. "
    "Generic, non-branded play props are fine and encouraged (building "
    "blocks, a play rug, cones, a basket of unrelated household items) "
    "since those clearly aren't part of what's being sold — the line to "
    "not cross is anything that reads as its own specific toy/product."
)

# A real "Learning" scene (kids at a table) came back with the actual photo
# letterboxed into a landscape strip with big blank white bars filling the
# rest of the canvas top and bottom. Likely cause: FRAMING_RULE's "leave
# empty space around the subject" was applied literally as blank canvas
# instead of more of the room/background — that instruction is about the
# PRODUCT having room to breathe within the photographed scene, not about
# the scene itself stopping short of the canvas edges.
FULL_BLEED_RULE = (
    "The photographed scene — background, room, floor, everything — must "
    "fill the ENTIRE image canvas edge to edge, with no blank white or "
    "empty bars/borders at the top, bottom, or sides (no letterboxing or "
    "pillarboxing). If you need empty space around the product for "
    "breathing room, that space must still be part of the photographed "
    "scene (more floor, more wall, more background) — never blank canvas "
    "outside the scene. Compose the shot so the actual photo content "
    "reaches all four edges of the image."
)

# The per-category slot descriptions are written for the WHOLE category
# (e.g. "Cars & RC Toys" covers both actual RC vehicles and plain die-cast
# cars), so a slot description can mention "the vehicle, controller, wheels,
# mechanism" even for a specific product that has no controller at all. The
# model was taking that literally and drawing a remote control next to a
# plain die-cast car. This keeps the slot description's wording generic
# while telling the model not to invent parts the real product doesn't have.
# A puzzle book's Feature images came back as broken crops of the flat
# cover artwork with the title text sliced off at every edge — the
# "zoom into a distinct part" instruction doesn't work when the reference
# image IS flat 2D print artwork with no separate 3D parts to zoom into.
FLAT_PRINT_RULE = (
    "This product is a flat printed item (a book, puzzle book, pad, or "
    "similar) — it has no separate mechanical parts or panels to zoom into. "
    "Do NOT crop into or zoom in on the cover artwork or page graphics — "
    "that only cuts off title text and art with no clear subject. Instead, "
    "photograph the ACTUAL PHYSICAL COPY as a real 3D object: for example "
    "the closed book/pad at a slight angle so its cover, spine, and page "
    "edges read as a real object sitting on a surface, or opened flat to "
    "show two full, uncropped inside pages, or held in a person's hands. "
    "Always keep the entire cover (or the entire open spread) inside the "
    "frame with visible margin on all sides — never crop into or past the "
    "edge of the artwork or any text."
)

NO_INVENTED_ACCESSORIES_RULE = (
    "Only depict parts and accessories that are clearly visible in the "
    "reference image or explicitly confirmed below. If the task wording "
    "above mentions something generic like a \"controller\" or \"remote\" "
    "but this specific product doesn't have one, do NOT add it."
)

# Same failure mode as the invented remote, but for VISUAL EFFECTS rather
# than physical parts: on a licensed character (e.g. a movie superhero
# figure), a generic "Action" slot description ("show it in an action
# pose") led the model to add fictional CGI-style effects the character
# has on-screen (glowing energy blasts, repulsor beams) that the physical
# toy does not actually have. This is a real ecommerce photo, not fan art
# or a movie still, unless the reference photo itself already shows the
# toy lit up that way (some figures DO have real LEDs).
NO_FICTIONAL_EFFECTS_RULE = (
    "Do not add glowing energy effects, light beams, sparks, fire, smoke, "
    "lightning, or motion-blur streaks unless the reference image already "
    "shows the product lit up or doing that — a character's on-screen "
    "movie powers are not real features of the physical toy. This is a "
    "realistic product photo, not fan art or a movie still."
)


# Some categories have multiple slots from the same "family" — Angle 1/2/3
# (up to 4 for Dolls & Doll House) or Feature 1/2/3 — whose rule text is
# just "show a different view/detail than the others", with no way for
# the model to know what its sibling slots actually produced (each is a
# separate API call). Real output showed two "Angle" slots for the same
# product come back as literally the same back-view shot. Assigning each
# a distinct CONCRETE instruction up front, deterministically by position,
# fixes this without needing the calls to see each other's output.
# Made these unambiguous and mutually exclusive after "opposite side
# (left/right)" got interpreted as ANOTHER rear view (a doll ended up with
# 3 of its 4 Angle slots all showing the back). Each one now names a
# camera position that cannot be confused with "the back".
ANGLE_ROTATION = [
    "the exact REAR view — the camera directly behind the product, showing "
    "its back. This must be the ONLY one of these images that shows the back "
    "— none of the other Angle images for this product may show the back view.",
    "a TOP-DOWN view — the camera positioned directly above the product "
    "looking straight down at its top surface. This is neither the front "
    "nor the back — it is looking down, not straight ahead.",
    "a tight close-up on ONE specific small detail (a texture, tag, face, "
    "logo, or joint) — zoomed in enough that the rest of the product is "
    "out of frame. This is NOT a full front or back view of the whole "
    "product at any distance.",
    "the LEFT SIDE profile — the camera positioned 90 degrees around from "
    "the front, showing only the left side of the product in profile. "
    "This is a SIDE view, not the front and not the back.",
]
FEATURE_ROTATION = [
    "a specific button, switch, or control and what it does",
    "a specific included accessory or attachment, shown attached or in use",
    "a specific material, texture, joint, or hinge/folding mechanism",
    "a specific compartment, storage space, or opening/closing mechanism",
]
SLOT_FAMILY_ROTATIONS = {"angle": ANGLE_ROTATION, "feature": FEATURE_ROTATION}


def assign_slot_variations(slots: dict, missing_slot_nums: set) -> dict:
    """{slot_num: concrete instruction} for slots that share a family with
    at least one other MISSING slot in this same product/run — see the
    rotation lists above. Slots in a family alone (nothing else missing
    with the same base name), or families we don't have a rotation for,
    get no entry — build_generation_prompt falls back to the slot's own
    rule text as before."""
    families = {}
    for slot_num in sorted(missing_slot_nums):
        if slot_num not in slots:
            continue
        family = re.sub(r"\s*\d+$", "", slots[slot_num]["image_type"]).strip().lower()
        families.setdefault(family, []).append(slot_num)

    assignments = {}
    for family, slot_nums in families.items():
        rotation = SLOT_FAMILY_ROTATIONS.get(family)
        if not rotation or len(slot_nums) < 2:
            continue
        for idx, slot_num in enumerate(slot_nums):
            assignments[slot_num] = rotation[idx % len(rotation)]
    return assignments


def build_generation_prompt(product_name: str, category: str, slot_info: dict,
                           description: str = "", specifications: str = "",
                           forced_variation: str = "") -> str:
    dimension_note = ""
    image_type_lower = slot_info["image_type"].lower()
    if "size" in image_type_lower or "dimension" in image_type_lower:
        # Specifications' "Dimensions (LxBxH)" wins over anything embedded
        # in the free-text description on conflict — see
        # merge_product_fields for why these can genuinely disagree.
        real_dims = extract_dimensions_from_description(description, specifications)
        if real_dims:
            # Without this, the model invents plausible-looking but wrong
            # numbers on the measurement lines — it has no way to know the
            # real size just from a photo. Real data beats a nice-looking guess.
            dimension_note = (
                f"\n\nIMPORTANT — real measurements: the manufacturer lists this "
                f"product's actual size as {real_dims}. Use these exact numbers on "
                f"the measurement lines and labels. Do not invent different numbers."
            )
        else:
            dimension_note = (
                f"\n\nNo exact manufacturer measurements were provided for this "
                f"product. Draw proportionate measurement lines based on the "
                f"reference image, and keep any numbers plausible for a product of "
                f"this type and apparent scale — do not present a guess as a "
                f"precise spec."
            )
        # Real output showed both a stray unlabeled arrow drawn straight
        # across the product's face, and the same measurement duplicated
        # as two separate arrows (two "Depth: 12 cm" callouts). Spelling
        # out the exact arrow count and where they may NOT go.
        axis_labels = AXIS_LABEL_RE.findall(real_dims)
        if axis_labels:
            is_vehicle = is_vehicle_product(category, product_name, description, specifications)
            arrow_labels, text_only_label = resolve_dimension_layout(axis_labels, is_vehicle)
            if text_only_label:
                # Structural fix, not another wording change — every
                # magnitude/wheel-landmark prompt tried on real vehicles
                # still swapped Length/Breadth. A strict side-profile shot
                # with only ONE horizontal arrow removes the judgment call
                # entirely: there's no second horizontal arrow left for the
                # model to place on the wrong edge, and Breadth isn't
                # visually measurable edge-on from a side profile anyway,
                # so it becomes a plain text spec line instead.
                length_label = next((l for l in arrow_labels if l.startswith("Length")), None)
                height_label = next((l for l in arrow_labels if l.startswith("Height")), "Height")
                n = len(arrow_labels)
                dimension_note += (
                    f"\n\nCamera angle — this is a wheeled vehicle: frame it in a "
                    f"strict SIDE-PROFILE view, camera positioned directly to the "
                    f"side and perpendicular to the vehicle's length (NOT a 3/4 "
                    f"angle, NOT a front/rear angle) — so the wheelbase and the "
                    f"vehicle's height are both visible edge-on, with no horizontal "
                    f"foreshortening to judge.\n\nArrow rules: draw EXACTLY {n} "
                    f"arrows on this entire image and nothing else — one for "
                    f"\"{length_label}\", running along the ground from the front "
                    f"wheel to the back wheel (the same side, nose to tail, the "
                    f"wheelbase), and one for \"{height_label}\", running vertically "
                    f"from the ground up to the vehicle's own top surface. Do NOT "
                    f"draw an arrow for \"{text_only_label}\" — from this "
                    f"side-profile angle it is not visible edge-on, so instead print "
                    f"\"{text_only_label}\" as a plain text label with no arrow and "
                    f"no line, sourced from the manufacturer measurement given "
                    f"above, placed in an empty corner of the image away from the "
                    f"two arrows. Never any additional, extra, or unlabeled arrow "
                    f"anywhere in the image for any other part or feature — if it "
                    f"isn't \"{length_label}\" or \"{height_label}\", it gets no "
                    f"arrow at all, no matter how prominent that part looks (e.g. a "
                    f"hanging rope, strap, cord, or handle is NOT one of these "
                    f"measurements, so it gets no arrow of its own). Every arrow and "
                    f"its label must start and end in the empty background space "
                    f"OUTSIDE the vehicle's outline, alongside it — none may cross, "
                    f"overlap, touch, or be drawn on top of the vehicle itself or "
                    f"anything attached to it."
                )
                axis_labels = arrow_labels
            else:
                # extract_dimensions_from_description already resolved which
                # raw number is which axis (Length/Breadth/Height, ...) when
                # the source key spelled out an order like "(LxBxH)" — a real
                # product still came back with only 2 of 3 arrows drawn and
                # the vertical one mislabeled with the wrong axis's number, so
                # this spells out each named measurement individually rather
                # than trusting the model to keep an unlabeled triple straight.
                n = len(axis_labels)
                horizontal_labels = [l for l in axis_labels if not l.startswith("Height")]
                magnitude_note = ""
                if len(horizontal_labels) >= 2:
                    ordered = sorted(horizontal_labels, key=axis_label_magnitude, reverse=True)
                    magnitude_note = (
                        f" Among the horizontal measurements, \"{ordered[0]}\" is the "
                        f"LARGEST number — its arrow must run along the visually LONGEST "
                        f"horizontal edge of the product. \"{ordered[-1]}\" is the "
                        f"SMALLEST — its arrow must run along the visually SHORTEST "
                        f"horizontal edge, perpendicular to the longest one. A real "
                        f"product came back with these two swapped (the bigger number "
                        f"attached to the visually shorter edge and vice versa) — match "
                        f"each arrow to its edge by actual visual proportion in the "
                        f"reference image, not by assuming which name conventionally "
                        f"goes where."
                    )
                    if is_vehicle:
                        magnitude_note += (
                            f" This is a wheeled vehicle: use the wheels as your "
                            f"landmark rather than judging apparent length. The "
                            f"wheels/axles run along the vehicle's real-world LENGTH "
                            f"(nose to tail) — so \"{ordered[0]}\" (the larger "
                            f"measurement) must be drawn along that same direction, "
                            f"front wheel to rear wheel. \"{ordered[-1]}\" (the "
                            f"smaller measurement) must be drawn across the narrow "
                            f"front or back face, perpendicular to the wheels — "
                            f"never along the wheelbase."
                        )
                    # Real output still swapped these even with the instructions
                    # above (both for a vehicle and for a plain box) — adding a
                    # concrete self-check step, not just another way of stating
                    # the same rule, since asking the model to verify its own
                    # draft before finalizing is a different mechanism than
                    # asking it to get the judgment right on the first pass.
                    magnitude_note += (
                        f" Self-check before finalizing this image: look at the two "
                        f"horizontal arrows you have actually drawn and compare their "
                        f"pixel lengths on the page. If the arrow labeled "
                        f"\"{ordered[-1]}\" (the smaller number) is drawn LONGER on "
                        f"the page than the arrow labeled \"{ordered[0]}\" (the "
                        f"larger number), that is backwards — swap which label is on "
                        f"which arrow (keep the arrows themselves where they are, "
                        f"just correct which text goes on which one) before producing "
                        f"the final image."
                    )
                composition_note = ""
                if n == 3:
                    # Structural fix: a real box product was shot flat-on
                    # (frontal), so its Height (a small number, e.g.
                    # 1.5 cm) was drawn as a tall vertical arrow the exact
                    # same on-screen length as Breadth (a much bigger
                    # number, e.g. 22 cm) — visually nonsensical even
                    # though both labels were individually correct. A
                    # correct real example used a 3/4 corner perspective
                    # (both ground-plane edges visible, receding from one
                    # near corner) where arrow lengths naturally track
                    # real proportions. Mandating that composition, plus
                    # an explicit proportionality rule, rather than just
                    # asking for correct labels.
                    composition_note = (
                        f" Camera angle: photograph this at a 3/4 CORNER "
                        f"perspective, not a flat frontal shot — position the "
                        f"camera so ONE bottom corner of the product is "
                        f"closest to the viewer, with both adjacent bottom "
                        f"edges receding away from that corner at an angle "
                        f"(like a classic product-dimension diagram, not a "
                        f"straight-on catalog photo). Draw the two horizontal "
                        f"arrows (Length and Breadth/Width) as diagonal arrows "
                        f"radiating from that same near corner, each running "
                        f"along its own visible ground-plane edge; draw the "
                        f"Height arrow as a separate vertical arrow at that "
                        f"same corner. Critically, each arrow's ON-SCREEN "
                        f"length must be visually proportional to its "
                        f"real-world number, consistent with the other arrows "
                        f"in this same image — the arrow for a larger number "
                        f"must look longer on the page than the arrow for a "
                        f"smaller number. A short real dimension (like a thin "
                        f"box's 1.5 cm height) must get a visibly SHORT arrow, "
                        f"never the same on-screen length as a much larger "
                        f"dimension just because both happen to be drawn "
                        f"vertically or from the same corner."
                    )
                dimension_note += (
                    f"\n\nArrow rules: draw EXACTLY {n} arrows on this entire image, one "
                    f"for each of these named measurements and nothing else — "
                    f"{', '.join(axis_labels)}. Never more than {n}, never fewer, never "
                    f"two arrows for the same measurement, and never any additional, "
                    f"extra, or unlabeled arrow anywhere in the image for any other part "
                    f"or feature — if it isn't one of these {n} named measurements, it "
                    f"gets no arrow at all, no matter how prominent that part looks (e.g. "
                    f"a hanging rope, strap, cord, handle, or other attachment is NOT one "
                    f"of the {n} measurements unless explicitly named above, so it gets no "
                    f"arrow of its own). Each arrow must run along the real-world axis its "
                    f"name describes (the Height arrow vertical along the product's actual "
                    f"height, the Length/Width/Breadth/Depth arrows along their own "
                    f"horizontal axes) — do not attach a label to the wrong axis or drop "
                    f"any of the {n} listed measurements.{magnitude_note}{composition_note} "
                    f"A base that is wider at the back than the front (a common "
                    f"perspective effect) still has only ONE length and ONE breadth — do "
                    f"not draw the same measurement a second time from the opposite corner "
                    f"or the far edge just because it is visible there too; pick ONE corner "
                    f"of the product and draw all {n} arrows radiating from measurements "
                    f"anchored at or near that single corner only. Every arrow and its "
                    f"label must start and end in the empty background space OUTSIDE the "
                    f"product's outline, alongside it — none may cross, overlap, touch, "
                    f"or be drawn on top of the product itself or anything attached to it."
                )
        else:
            dimension_note += (
                "\n\nArrow rules: draw exactly one arrow per dimension given above "
                "(so 2 arrows for a 2D size, 3 for length/width/height) — never more, "
                "and never two arrows for the same measurement. Every arrow and its "
                "label must sit in the empty space OUTSIDE the product's outline, "
                "alongside it — none may cross, overlap, or be drawn on top of the "
                "product itself. Do not add any extra arrow or line beyond these."
            )
        # Requested on every dimension image, matching how real manufacturer
        # spec-sheet photos commonly caveat printed measurements — covers
        # small, expected manufacturing/rendering tolerance without it
        # reading as a wrong-number defect.
        dimension_note += (
            "\n\nAlso add the small text \"Size may vary slightly\" once, in a "
            "plain small gray font clearly smaller than the measurement "
            "labels, in an empty corner of the image (e.g. bottom-right). It "
            "must not overlap any arrow, measurement label, or the product "
            "itself."
        )
    # Every slot type: keep the whole product in frame — cropping at the
    # edges (e.g. a close-up "Feature" shot clipping the wheel) has shown up
    # in real output. Lifestyle specifically: no background blur/bokeh —
    # requested explicitly, and a blurred background reads as lower catalog
    # quality even though it's a common lifestyle-photography convention.
    extra_rules = FRAMING_RULE + " " + NO_INVENTED_ACCESSORIES_RULE + " " + NO_FICTIONAL_EFFECTS_RULE
    # Any slot that composes a real-world scene (not just the product on a
    # plain studio background) is where letterboxing showed up — plain
    # product shots on white are unaffected since "empty space around the
    # product" IS the correct white background there.
    if any(k in image_type_lower for k in ("lifestyle", "learning", "skills", "action")):
        extra_rules += " " + FULL_BLEED_RULE
    if "lifestyle" in image_type_lower:
        extra_rules += " " + FOCUS_RULE + " " + INDIA_REPRESENTATION_RULE + " " + NO_OTHER_PRODUCTS_RULE
        # A small RC car came back looking oversized next to the child
        # holding it — a Lifestyle shot has no reference-image anchor for
        # scale the way a straight product photo does (the model is
        # composing a whole new scene), so it needs an explicit real-world
        # size check against the people in it.
        lifestyle_dims = extract_dimensions_from_description(description, specifications)
        if lifestyle_dims:
            extra_rules += (
                f" Scale check: the manufacturer lists this product's actual size "
                f"as {lifestyle_dims}. Render it at a size, relative to the "
                f"people in this scene, that is genuinely consistent with those "
                f"real dimensions — not larger or smaller than a product that "
                f"size would actually look in someone's hands or on the floor."
            )
        else:
            extra_rules += (
                " Scale check: no exact size was given for this product, but "
                "keep it rendered at a size that is realistic for a product "
                "matching this description relative to the people in the scene "
                "— do not enlarge or shrink it to be more dramatic or eye-catching "
                "than the real product actually is."
            )
    if "feature" in image_type_lower:
        flat_print_keywords = (
            "book", "workbook", "notebook", "flash card", "flashcard",
            "sticker book", "colouring", "coloring", "puzzle book",
            "activity pad", "activity book",
        )
        # Specifications' "Contents" field is often the clearest signal
        # (e.g. "Contents: 1 Story Book") even when neither the product
        # name nor the free-text description literally says "book".
        haystack = f"{product_name} {description} {specifications}".lower()
        if any(kw in haystack for kw in flat_print_keywords):
            extra_rules += " " + FLAT_PRINT_RULE
        else:
            # Telling the model not to crop wasn't enough on its own — for a
            # multi-part product (many small pieces/panels) it kept reaching
            # for a scattered flat-lay of many items, and packing that many
            # separate things in inevitably ran individual pieces off the
            # edge. This targets the actual cause: fewer items, more room.
            extra_rules += " " + FEATURE_STYLE_RULE
        extra_rules += " " + FEATURE_TEXT_LABEL_RULE
        if is_vehicle_product(category, product_name, description, specifications):
            # Structural fix, not another wording tweak — asking the model
            # to pinpoint a small coordinate for a wheel-related label
            # inside a full-vehicle shot has now failed 4 different ways
            # across real vehicles (pointer landed on: a mixer drum, a
            # window/pillar, a bumper/fender edge, a hood/headlight), each
            # time after the previous exact failure was spelled out and
            # forbidden by name. Reframing the ENTIRE shot as a close-up
            # crop on one wheel — so the wheel is the dominant round object
            # filling the frame, not a small part of a wide scene — removes
            # the coordinate-guessing problem instead of asking for more
            # precision at it.
            extra_rules += (
                " If the feature you choose to highlight relates to the "
                "wheels, axle, or rolling/suspension mechanism: do NOT shoot "
                "a full side or angled view of the whole vehicle for this "
                "image. Instead, frame this ENTIRE shot as a tight close-up "
                "crop centered on ONE wheel, with that wheel's tire and rim "
                "filling at least 40% of the frame width — the rest of the "
                "vehicle may be partially visible at the edges, softly out "
                "of focus, or cropped off entirely, since this is a "
                "close-up feature shot, not a full product shot. With the "
                "wheel this large and dominant, there is no other round "
                "part nearby it could be confused with — the label's "
                "pointer/leader line must land on that same large wheel, "
                "on the visible tire or rim itself, not on the wheel arch, "
                "fender, bumper, hood, or any body panel at the edge of the "
                "crop."
            )
    if any(k in image_type_lower for k in ("angle", "second", "rear", "back", "opposite")):
        extra_rules += " " + PACKAGING_LOGIC_RULE

    accessory_note = ""
    if is_battery_operated(description, specifications) is False:
        # A confirmed "No" is stronger than the generic rule above — say it
        # explicitly rather than relying on the model inferring it from a
        # photo that may not make battery/RC absence obvious at all.
        accessory_note = (
            "\n\nThis product has NO battery, NO remote control, and NO "
            "controller — do not depict any of those in this image, even if "
            "the task wording above mentions one generically."
        )

    # General product context beyond the two specific things we parse out
    # (dimensions, battery) — the per-category slot wording is written for
    # the whole category, not this specific product, so giving the model
    # the actual description helps it avoid inventing anything else
    # inconsistent with what this product really is (material, whether it's
    # interactive, what's actually included, etc.), the same class of bug
    # as the invented remote control.
    context_note = ""
    if description or specifications:
        spec_block = f"\nSpecification section: {specifications}" if specifications else ""
        context_note = (
            f"\n\nProduct facts (use this to understand what the product "
            f"actually is — do not depict anything that contradicts it):\n"
            f"{description[:800]}{spec_block}"
        )

    variation_note = ""
    if forced_variation:
        # Overrides the slot's own vague "show a different angle/detail"
        # wording with something concrete and specific to this slot, so two
        # sibling "Angle"/"Feature" slots generated in separate API calls
        # can't both land on the same obvious choice (e.g. both picking the
        # back view).
        variation_note = (
            f"\n\nSPECIFIC REQUIREMENT for this image: it must show {forced_variation}. "
            f"This is what makes it different from the other images generated for "
            f"this product — do not default to a generic or repeated angle."
        )

    # "No added text" is the right default everywhere else, but it directly
    # contradicts FEATURE_TEXT_LABEL_RULE above for Feature-type slots,
    # which requires exactly one short real caption.
    quality_note = ("high resolution, realistic photography, clean "
                    "professional catalog style, no watermark")
    if "feature" in image_type_lower:
        quality_note += (
            ". The one short feature caption required above is the ONLY "
            "text allowed on this image — no other added text or watermark."
        )
    else:
        quality_note += ", no added text."

    return (
        f"You are editing an ecommerce product photo for a toy called "
        f"\"{product_name}\" (category: {category}).\n\n"
        f"STRICT RULE: {LOCK_RULES}"
        f"{context_note}\n\n"
        f"Task — generate this specific image type: {slot_info['image_type']}.\n"
        f"Requirement: {slot_info['description']}"
        f"{dimension_note}{accessory_note}{variation_note}\n\n"
        f"Composition rules: {extra_rules}\n\n"
        f"Ecommerce quality: {quality_note}"
    )


def upscale_to_minimum(image_bytes: bytes, min_px: int) -> bytes:
    """If the model's output is smaller than min_px on its shorter side,
    upscale it with high-quality Lanczos resampling so the saved file meets
    the resolution floor. This is genuinely an upscale, not new detail —
    it makes the file "at least Nx N pixels", not sharper than the model's
    native output. Left as-is if already >= min_px (never downscales)."""
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    if min(width, height) >= min_px:
        return image_bytes

    scale = min_px / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    upscaled = img.resize(new_size, Image.LANCZOS)

    buf = BytesIO()
    upscaled.save(buf, format=img.format or "PNG")
    return buf.getvalue()


# Classification's MIN_EXISTING_IMAGE_PX (500px) only rejects genuinely
# unusable photos — it let through real admin-panel images anywhere from
# 500px up to a few thousand, well below the 2048px ("2K") catalog floor
# we hold AI-generated output to. An "Existing" slot with a 700x700 photo
# was being treated as fully covered with no quality check at all past
# that low floor. This re-checks every Existing slot against the same
# 2048px bar and, if it falls short, upscales it (never fabricates new
# detail) and re-hosts the upscaled copy so the sheet links to a file that
# actually meets the floor instead of the low-res original.
def ensure_existing_image_quality(url: str, filename: str, out_dir: str,
                                  uploader: "GcsUploader | None",
                                  upload_cache: "UploadCache",
                                  upload_failures: int) -> dict:
    """Returns {"status": "Existing" | "Existing (enhanced)" | "Existing (quality_check_failed)",
    "image_source": ..., "gcp_link": ..., "upload_failures": ...}."""
    try:
        content, _ = get_image_bytes(url)
        width, height = Image.open(BytesIO(content)).size
    except Exception:
        # Can't verify it — don't block the pipeline on this, just pass the
        # original through unchanged and note that quality wasn't verified.
        return {"status": "Existing (quality_check_failed)", "image_source": url,
                "gcp_link": "", "upload_failures": upload_failures}

    if min(width, height) >= MIN_OUTPUT_PX:
        return {"status": "Existing", "image_source": url, "gcp_link": "",
                "upload_failures": upload_failures}

    enhanced = upscale_to_minimum(content, MIN_OUTPUT_PX)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "wb") as f:
        f.write(enhanced)
    gcp_link, upload_failures = maybe_upload(uploader, upload_cache, filename,
                                             out_path, upload_failures)
    return {"status": "Existing (enhanced)", "image_source": filename,
            "gcp_link": gcp_link, "upload_failures": upload_failures}


def pick_reference_image(covered: dict, image_urls: list) -> str:
    """Prefer the front view (slot 1), then any classified image, then
    whatever the product has — the edit needs the real toy to preserve."""
    if 1 in covered:
        return covered[1]
    if covered:
        return covered[min(covered)]
    return image_urls[0] if image_urls else ""


def generate_image(reference_url: str, prompt: str, out_path: str,
                   project_id: str, region: str, tokens: VertexTokenProvider) -> dict:
    try:
        ref_bytes, content_type = get_image_bytes(reference_url)
    except (requests.RequestException, OSError) as e:
        return {"status": f"reference_download_failed: {e}"}

    ref_b64 = base64.b64encode(ref_bytes).decode()

    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{MODEL_ID}:generateContent"
    )
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": content_type, "data": ref_b64}},
                {"text": prompt},
            ],
        }],
        # imageConfig.imageSize is a no-op on gemini-2.5-flash-image (verified
        # empirically — output stays 1024x1024 regardless), but harmless to
        # send in case a model that honors it becomes available later. The
        # MIN_OUTPUT_PX upscale below is what actually guarantees the floor.
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                             "imageConfig": {"imageSize": "2K"}},
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        headers = {"Authorization": f"Bearer {tokens.token()}",
                   "Content-Type": "application/json"}
        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                candidates = result.get("candidates") or []
                if not candidates:
                    # Usually a safety block on the prompt or reference image.
                    reason = result.get("promptFeedback", {}).get("blockReason", "unknown")
                    return {"status": f"no_candidates: {reason}"}
                parts = candidates[0].get("content", {}).get("parts") or []
                image_part = next((p for p in parts if "inlineData" in p), None)
                if not image_part:
                    finish = candidates[0].get("finishReason", "")
                    return {"status": f"no_image_in_response: {finish or 'no reason given'}"}
                image_bytes = base64.b64decode(image_part["inlineData"]["data"])
                image_bytes = upscale_to_minimum(image_bytes, MIN_OUTPUT_PX)
                with open(out_path, "wb") as f:
                    f.write(image_bytes)
                return {"status": "generated"}

            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in RETRYABLE_STATUS:
                return {"status": f"failed: {last_error}"}
            if resp.status_code == 401:
                tokens.token(force_refresh=True)
        except requests.RequestException as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            time.sleep(2 * attempt)

    return {"status": f"failed: {last_error}"}


# Non-capturing group is deliberate: .findall() with exactly one capturing
# group returns ONLY that group's text, not the full match — which silently
# turned every axis_labels entry into a bare name ("Length") instead of the
# full "Length 7.5 cm", dropping the numbers everywhere this list is used
# (the arrow-count enumeration, and the magnitude-matching check below).
AXIS_LABEL_RE = re.compile(r"(?:Length|Width|Breadth|Height|Depth) [\d.]+\s*[a-zA-Z]*")


def axis_label_magnitude(label: str) -> float:
    """Numeric value out of a label like "Length 7.5 cm" -> 7.5. Shared by
    build_generation_prompt (to tell the model which named measurement
    should land on the visually longer vs shorter edge) and
    verify_dimension_image (to check it actually did)."""
    m = re.search(r"[\d.]+", label)
    return float(m.group()) if m else 0.0


VEHICLE_KEYWORDS = ("cars & rc", "rc toy", "ride-on", "ride on", "tricycle", "wheel")


def is_vehicle_product(category: str, product_name: str, description: str,
                       specifications: str) -> bool:
    """Whether the wheel-landmark dimension rule applies — shared by
    build_generation_prompt and process_slot_task (which passes the result
    to generate_image_with_verification) so both use the exact same
    detection."""
    haystack = f"{category} {product_name} {description} {specifications}".lower()
    return any(k in haystack for k in VEHICLE_KEYWORDS)


def resolve_dimension_layout(axis_labels: list, is_vehicle: bool) -> tuple:
    """For vehicles, collapses the arrow layout from 3 down to 2: keep
    Length (front wheel to back wheel) and Height (ground to top surface)
    as drawn arrows, and push Breadth to a plain text-only label with no
    arrow of its own.

    This is a structural fix, not another wording change — prompt-only
    instructions (magnitude comparison, then vague wheel landmarks, then
    exact wheel-to-wheel landmarks) all still produced a Length/Breadth
    swap on real vehicles across multiple attempts, including when the
    model was told to self-check its own arrow lengths before finalizing.
    Reframing to a side-profile shot with only one horizontal arrow removes
    the judgment call entirely — there is no second horizontal arrow left
    for the model to place on the wrong edge, and Breadth isn't visually
    measurable edge-on from a side profile anyway.

    Returns (arrow_labels, text_only_label). For non-vehicles, or a vehicle
    missing a clear Length/Breadth pair, arrow_labels is axis_labels
    unchanged and text_only_label is None.
    """
    if not is_vehicle:
        return axis_labels, None
    length_label = next((l for l in axis_labels if l.startswith("Length")), None)
    breadth_label = next((l for l in axis_labels
                          if l.startswith("Breadth") or l.startswith("Width")), None)
    if not length_label or not breadth_label:
        return axis_labels, None
    arrow_labels = [l for l in axis_labels if l != breadth_label]
    return arrow_labels, breadth_label


def compute_axis_labels(description: str, specifications: str) -> list:
    """The same axis-name extraction used inside build_generation_prompt's
    dimension_note, exposed standalone so the caller can decide up front
    whether this slot has a checkable ground truth (a known, named set of
    measurements) worth verifying after generation — see
    verify_dimension_image. Empty list if the source data has no
    LxBxH-style axis-order hint (nothing to verify against)."""
    real_dims = extract_dimensions_from_description(description, specifications)
    return AXIS_LABEL_RE.findall(real_dims)


# Vertex AI vision model for the verify step — a text+vision model like the
# classify step uses, NOT the image-generation model above.
DIMENSION_VERIFY_MODEL = os.environ.get("GEMINI_CLASSIFY_MODEL", "gemini-2.5-flash")

DIMENSION_VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        # Forcing an enumeration of every single arrow BEFORE the model
        # commits to a valid/invalid verdict catches things a single
        # holistic glance misses — a real check missed 2 short unlabeled
        # arrows crossing the product's interior when only asked for a
        # final yes/no; making it list each one by location first (a
        # cheap form of forced attention) is meaningfully more reliable.
        "arrows_found": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": ("One entry per MEASUREMENT arrow we added anywhere in the "
                           "image, however small or short — format: '<label or "
                           "UNLABELED>: <exact location, e.g. \"both endpoints in "
                           "blank background to the right\" or \"crosses the red seat "
                           "in the center\">'. Do NOT include arrows, curved lines, or "
                           "icons that are printed as part of the product's OWN "
                           "packaging artwork/logo (e.g. a decorative arrow icon "
                           "printed on the box itself) — a real check flagged a valid "
                           "image as broken because it mistook the box's own printed "
                           "logo arrow for an extra measurement arrow. Only count "
                           "arrows that were added on top of the product photo as a "
                           "measurement callout."),
        },
        # A separate, purely perceptual field rather than folding this into
        # the model's own "valid" judgment — a real check reported valid=
        # true on an image where the bigger number was visibly on the
        # shorter edge, because combining "perceive which edge is longer"
        # and "apply the swap-check logic" in one holistic judgment let the
        # logic step silently fail even when the raw perception would have
        # been fine. Asking only for the observable fact here, then
        # comparing it against the expected order in plain Python (see
        # verify_dimension_image), is more reliable than trusting the
        # model to also get the comparison right.
        "longest_horizontal_arrow_label": {
            "type": "STRING",
            "description": ("Which named horizontal measurement's arrow is drawn "
                           "visually LONGEST on the page (ignore Height) — just the "
                           "name, e.g. 'Length'. Look at actual on-screen arrow "
                           "length, not which number is bigger."),
        },
        # For wheeled vehicles specifically: "visually longest" turned out
        # to be an unreliable judgment call under 3/4-angle foreshortening
        # (a real check still got this backwards even when asked directly).
        # Anchoring to the wheels — a concrete, unambiguous landmark that
        # doesn't require any perspective judgment — is more reliable.
        "wheel_direction_arrow_label": {
            "type": "STRING",
            "description": ("Only for a wheeled vehicle product: which named "
                           "horizontal measurement's arrow runs from the front "
                           "wheel to the back wheel (along the wheelbase, same "
                           "side, nose to tail)? For a vehicle this should be "
                           "\"Length\" by definition. Just the name. Leave "
                           "blank if this product has no wheels."),
        },
        # Real output rendered "Breadeth" instead of "Breadth" (and, in an
        # earlier image, "Heglt" instead of "Height") — a spelling glitch
        # in the model's own text rendering, not a prompt-wording problem,
        # since the correct spelling is what we asked for. Transcribing
        # each label's name character-for-character (not auto-corrected)
        # lets us catch this deterministically in code, the same pattern
        # used for the magnitude/wheel checks above.
        "label_name_texts": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": ("The NAME word of EVERY text label in the image (Length, "
                           "Breadth, Height, Width, Depth, etc.) — including any label "
                           "that has no arrow of its own, just plain text. Transcribe "
                           "each one EXACTLY as it is spelled/rendered in the image, "
                           "character for character, even if it looks misspelled — do "
                           "NOT auto-correct it to what you think it was supposed to "
                           "say. One entry per label."),
        },
        "valid": {"type": "BOOLEAN"},
        "arrow_count": {"type": "INTEGER"},
        "reason": {"type": "STRING"},
    },
    "required": ["arrows_found", "valid", "arrow_count", "reason"],
}


def verify_dimension_image(image_bytes: bytes, axis_labels: list, project_id: str,
                          region: str, tokens: VertexTokenProvider,
                          is_vehicle: bool = False, text_only_label: str = "") -> dict:
    """Checks a generated dimension image against its own ground truth — the
    exact set of named measurements it was asked to draw — instead of
    trusting the generation call got it right. Real output repeatedly came
    back with a duplicated or dropped measurement, a stray unlabeled arrow,
    or an arrow drawn across the product, even after the prompt spelled out
    the exact count and names. This is what makes the retry loop in
    generate_image_with_verification meaningful rather than blind luck.

    Fails OPEN (valid=True) on any error calling the verifier itself — a
    flaky verification call should not burn through the retry budget or
    block the row; the image just goes out unverified (flagged in Status)
    rather than blocking the whole run.
    """
    names = [label.split()[0] for label in axis_labels]
    n = len(names)
    text_only_name = text_only_label.split()[0] if text_only_label else None
    expected_spelling_names = names + ([text_only_name] if text_only_name else [])

    horizontal_labels = [l for l in axis_labels if not l.startswith("Height")]
    expected_longest_name = None
    expect_length_on_wheels = False
    magnitude_check = ""
    if len(horizontal_labels) >= 2:
        ordered = sorted(horizontal_labels, key=axis_label_magnitude, reverse=True)
        expected_longest_name = ordered[0].split()[0]
        magnitude_check = (
            f" Separately, report which of the horizontal measurements' arrows "
            f"is drawn visually longest on the page in longest_horizontal_arrow_label "
            f"— judge this purely by looking at actual on-screen arrow length, not "
            f"by assuming which name should be longer."
        )
        length_label = next((l for l in horizontal_labels if l.startswith("Length")), None)
        if is_vehicle and length_label:
            # Name-based, not magnitude-based: "Length" must be the one
            # anchored to the wheels, by definition, regardless of which
            # number happens to be bigger — this is what actually held up
            # in generation (see build_generation_prompt) after magnitude
            # comparisons alone kept getting swapped on real vehicles.
            expect_length_on_wheels = True
            magnitude_check += (
                f" This product is a wheeled vehicle, so ALSO report in "
                f"wheel_direction_arrow_label which named measurement's arrow runs "
                f"from the front wheel to the back wheel (along the wheelbase, "
                f"same side, nose to tail) — this should be \"Length\" by "
                f"definition for a vehicle. Judge this by the wheels' physical "
                f"position, not by which arrow looks longer on screen — a real "
                f"check got this backwards even when comparing apparent length."
            )
        elif is_vehicle:
            magnitude_check += (
                f" This product is a wheeled vehicle, so ALSO report in "
                f"wheel_direction_arrow_label which named measurement's arrow runs "
                f"in the same direction as the wheels (front wheel to rear wheel) "
                f"— this is a more reliable check than apparent length for a "
                f"vehicle shot at an angle, since foreshortening can make the "
                f"true longer edge look shorter on screen."
            )

    text_only_note = (
        f" The image should also show \"{text_only_label}\" as a plain text "
        f"label with no arrow of its own — include its name in "
        f"label_name_texts too."
        if text_only_label else ""
    )
    prompt = (
        f"This product image is supposed to show exactly {n} measurement "
        f"arrows for these named dimensions, each appearing exactly once: "
        f"{', '.join(names)}.{text_only_note} "
        f"First, scan the ENTIRE image very carefully — including the "
        f"interior and center of the product, not just its outer edges "
        f"and background — and list EVERY measurement arrow or double-headed "
        f"line you can find in arrows_found, no matter how short, thin, or "
        f"easy to miss at a glance; a short unlabeled arrow crossing the "
        f"middle of the product is a real, common defect here and must not "
        f"be overlooked. IMPORTANT — do not confuse this with a decorative "
        f"arrow, swoosh, or icon that is part of the product's OWN printed "
        f"packaging design/logo (e.g. a printed arrow icon on the box "
        f"itself, already present before any measurement was added) — a "
        f"real check wrongly flagged a correct image as broken because it "
        f"mistook the box's own printed logo arrow for an extra measurement "
        f"arrow; only count arrows that were added ON TOP of the product "
        f"photo as a callout. Then, using that list, check all of the following: "
        f"(1) there are EXACTLY {n} arrows total, not more, not fewer; "
        f"(2) each of {', '.join(names)} appears exactly once — none "
        f"missing, none duplicated (e.g. two arrows both for the same "
        f"named measurement, even if one is unlabeled, counts as a "
        f"duplicate); (3) no arrow crosses, overlaps, or touches the "
        f"product itself or anything attached to it (like a rope, strap, "
        f"or handle) — every arrow must lie entirely in empty background "
        f"space; (4) fill in label_name_texts with the EXACT spelling of "
        f"every text label's name in the image, transcribed character for "
        f"character even if it looks misspelled — do not silently "
        f"auto-correct a typo when transcribing it. Set valid=true only if "
        f"(1)-(3) hold for every arrow in arrows_found.{magnitude_check}"
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{DIMENSION_VERIFY_MODEL}:generateContent"
    )
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image_bytes).decode()}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": DIMENSION_VERIFY_SCHEMA},
    }
    try:
        headers = {"Authorization": f"Bearer {tokens.token()}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            return {"valid": True, "reason": f"verification_error: HTTP {resp.status_code}"}
        candidates = resp.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") or [] if candidates else []
        text = next((p["text"] for p in parts if "text" in p), None)
        if text is None:
            return {"valid": True, "reason": "verification_error: no_text_in_response"}
        parsed = json.loads(text)
        valid = bool(parsed.get("valid"))
        reason = parsed.get("reason", "")
        # Deterministic override: don't trust the model's own combined
        # judgment for the swap-check — compare its raw perceptual report
        # against the expected order in plain code. A real check reported
        # valid=true while the bigger number sat on the visually shorter
        # edge, because folding "perceive" + "apply this specific logic"
        # into one holistic verdict let the logic silently fail even when
        # the underlying perception (if asked for directly) would show it.
        if expected_longest_name:
            wheel_reported = (parsed.get("wheel_direction_arrow_label") or "").strip()
            if is_vehicle and expect_length_on_wheels and wheel_reported:
                # Name-based check: "Length" must be the one on the wheels,
                # by definition, regardless of magnitude — matches the
                # generation-side instruction (see build_generation_prompt).
                if wheel_reported.split()[0].lower() != "length":
                    valid = False
                    reason = (f"axis_swap_detected: model reported '{wheel_reported}' as "
                             f"running along the wheels (front-to-back), but 'Length' is "
                             f"defined as that measurement for a vehicle regardless of "
                             f"which number is bigger. ({reason})")
            elif is_vehicle and wheel_reported:
                if wheel_reported.split()[0].lower() != expected_longest_name.lower():
                    valid = False
                    reason = (f"axis_swap_detected: model reported '{wheel_reported}' as "
                             f"running along the wheels, but '{expected_longest_name}' has "
                             f"the larger number and should align with the wheels. ({reason})")
            else:
                reported = (parsed.get("longest_horizontal_arrow_label") or "").strip()
                if reported and reported.split()[0].lower() != expected_longest_name.lower():
                    valid = False
                    reason = (f"axis_swap_detected: model reported '{reported}' as the "
                             f"visually longest horizontal arrow, but '{expected_longest_name}' "
                             f"has the larger number and should be longest. ({reason})")
        # Deterministic spelling check — real output rendered "Breadeth"
        # for "Breadth" and "Heglt" for "Height", both passed by the
        # verifier because it was never asked to check spelling at all.
        # Comparing the model's own (not-auto-corrected) transcription
        # against the exact expected name in code catches this the same
        # way the swap-check above catches a bad holistic judgment.
        reported_names_lower = [
            (t or "").strip().lower() for t in (parsed.get("label_name_texts") or [])
        ]
        for expected_name in expected_spelling_names:
            if expected_name.lower() not in reported_names_lower:
                valid = False
                reason = (f"spelling_error: expected a label spelled '{expected_name}' "
                         f"but it was not found among the transcribed labels "
                         f"{parsed.get('label_name_texts')!r} — likely misspelled or "
                         f"missing in the image. ({reason})")
        return {"valid": valid, "reason": reason}
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        return {"valid": True, "reason": f"verification_error: {e}"}


MAX_GENERATION_ATTEMPTS = 3


def generate_image_with_verification(reference_url: str, prompt: str, out_path: str,
                                     project_id: str, region: str, tokens: VertexTokenProvider,
                                     axis_labels: list, is_vehicle: bool = False,
                                     text_only_label: str = "") -> dict:
    """Wraps generate_image with the verify-and-retry loop: for a slot with
    checkable ground truth (axis_labels non-empty), regenerate up to
    MAX_GENERATION_ATTEMPTS (3, capped — real cost per attempt) total times
    until verify_dimension_image passes. Exhausting all 3 without a pass
    keeps the LAST attempt's file (better than nothing) but reports
    "generated_unverified" so it can be flagged for manual review instead
    of silently shipped as a clean pass.

    Slots with no axis_labels (no checkable ground truth) generate once,
    exactly as before — no extra verification cost where we can't actually
    verify anything meaningful.
    """
    last_result = None
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        result = generate_image(reference_url, prompt, out_path, project_id, region, tokens)
        if result["status"] != "generated":
            last_result = result
            continue
        if not axis_labels:
            return result
        with open(out_path, "rb") as f:
            image_bytes = f.read()
        verdict = verify_dimension_image(image_bytes, axis_labels, project_id, region, tokens,
                                        is_vehicle=is_vehicle, text_only_label=text_only_label)
        if verdict["valid"]:
            return result
        last_result = {"status": f"generated_unverified: {verdict['reason']}"}
    # Ran out of attempts — last_result is either a real generation failure
    # or "generated_unverified" (file on disk, just never passed the check).
    return last_result


def build_gcs_uploader() -> "GcsUploader | None":
    """None if GCS isn't configured — auto-upload is opt-in, everything
    still works with a manual upload as before."""
    bucket_env = os.environ.get("GCS_BUCKET")
    if not bucket_env:
        return None

    key_path = os.environ.get("GCS_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        sys.exit("ERROR: GCS_BUCKET is set but no GCS_CREDENTIALS (or "
                 "GOOGLE_APPLICATION_CREDENTIALS) is set for it.")

    bucket, _, embedded_prefix = bucket_env.partition("/")
    prefix = "/".join(p for p in (embedded_prefix, os.environ.get("GCS_PREFIX", "")) if p).strip("/")
    return GcsUploader(key_path, bucket, prefix)


class UploadCache:
    """Filename -> {content hash, public URL}, persisted next to the
    generated images so a re-run doesn't re-upload files that already made
    it to GCS UNCHANGED.

    Keyed on content hash, not just filename: if a file gets regenerated
    with different bytes under the same filename (e.g. after fixing a
    prompt and forcing that one slot to regenerate), the old cache entry's
    hash won't match the new file's hash, so it re-uploads instead of
    silently serving the stale link. Caching on filename alone was a real
    bug — a corrected image was generated locally but the old GCS object
    (and its URL) never got replaced because the cache still remembered
    the previous upload for that filename.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            raw = {}
        # Older cache files (pre content-hash) stored filename -> plain URL
        # string. Treat those as "no hash on record" rather than crashing —
        # they'll just re-upload once and get a proper entry going forward.
        self._data = {k: v for k, v in raw.items() if isinstance(v, dict)}

    def get(self, filename: str, content_hash: str) -> str:
        entry = self._data.get(filename)
        if entry and entry.get("hash") == content_hash:
            return entry.get("url", "")
        return ""

    def set(self, filename: str, content_hash: str, url: str):
        # set() rewrites the WHOLE file from self._data each call. Without
        # a lock, concurrent workers (see run_concurrent in main()) racing
        # this read-modify-write can silently lose each other's entries —
        # thread A's slower write finishes after thread B's and overwrites
        # the file with a snapshot missing B's entry entirely. Not
        # corruption (each individual write is still valid JSON), just a
        # lost cache entry that causes a needless re-upload next run.
        with self._lock:
            self._data[filename] = {"hash": content_hash, "url": url}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)


def maybe_upload(uploader: "GcsUploader | None", cache: UploadCache,
                 filename: str, out_path: str, upload_failures: int) -> tuple:
    """Returns (gcp_link, updated_upload_failures). No-op ("", unchanged
    count) when GCS isn't configured. A single failed upload is logged and
    counted, not fatal — the run keeps going and that row's GCP_Link is
    left blank for manual upload."""
    if not uploader:
        return "", upload_failures

    with open(out_path, "rb") as f:
        data = f.read()
    content_hash = hashlib.md5(data).hexdigest()

    cached = cache.get(filename, content_hash)
    if cached:
        return cached, upload_failures

    try:
        url = uploader.upload(data, filename, "image/png")
        cache.set(filename, content_hash, url)
        return url, upload_failures
    except GcsUploadError as e:
        print(f"  GCS upload failed for {filename}: {e}")
        return "", upload_failures + 1


def load_inputs(products_path: str, classification_path: str) -> pd.DataFrame:
    """Join the fetch and classify outputs.

    Joined on Product_ID alone: SKU and category come from the same fetch
    step on both sides, so including them as keys only creates a way for a
    product to silently drop out of the join.
    """
    products = pd.read_csv(products_path).fillna("")
    classification = pd.read_csv(classification_path).fillna("")

    products["Product_ID"] = products["Product_ID"].astype(str)
    classification["Product_ID"] = classification["Product_ID"].astype(str)

    carry = ["Product_ID", "Status", "Missing_Slots", "Covered_Slots"]
    if "Rule_Category" in classification.columns:
        carry.append("Rule_Category")
    missing_cols = [c for c in carry if c not in classification.columns]
    if missing_cols:
        sys.exit(f"ERROR: {classification_path} is missing expected column(s) "
                 f"{missing_cols}. Re-run classify_images.py to regenerate it.")

    merged = products.merge(
        classification[carry].drop_duplicates(subset="Product_ID"),
        on="Product_ID", how="left",
    ).fillna("")

    unclassified = (merged["Status"] == "").sum()
    if unclassified:
        print(f"WARNING: {unclassified} of {len(merged)} products have no row in "
              f"{classification_path} — they will be reported as "
              f"unknown_not_classified. Re-run step 2 over the full product list "
              f"to cover them.")
    return merged


def build_slot_tasks(merged: pd.DataFrame, rules: RuleMaster, image_out_dir: str) -> list:
    """Flattens every (product, slot) into an independent task dict up
    front — no API calls yet, just the bookkeeping needed to process one
    slot in isolation. This is what makes the slots safe to hand to a
    thread pool: each task only reads its own row's data, never touches
    another task's state.
    """
    tasks = []
    for _, row in merged.iterrows():
        base_row = {"Product_ID": row["Product_ID"], "SKU": row["SKU"],
                   "Name": row["Name"], "Category": row.get("Category_L1", ""),
                   "Slot": "", "Image_Type": "", "Status": "",
                   "Image_Source": "", "GCP_Link": ""}

        rule_category, slots = rules.match(
            row.get("Rule_Category", ""), row.get("Category_L1", ""),
            row.get("Category_L2", ""), row.get("Category_L3", ""),
        )
        if not slots:
            tasks.append({"key": f"{row['Product_ID']}:no_rule", "kind": "no_rule",
                         "row_base": base_row})
            continue

        covered = parse_slot_map(row.get("Covered_Slots", ""))
        missing_slot_nums = set(parse_slot_map(row.get("Missing_Slots", "")))
        reference_url = pick_reference_image(covered, split_image_urls(row.get("Image_URLs", "")))
        slot_variations = assign_slot_variations(slots, missing_slot_nums)

        for slot_num in sorted(slots):
            slot_info = slots[slot_num]
            tasks.append({
                "key": f"{row['Product_ID']}:{slot_num}",
                "kind": "slot",
                "row_base": {**base_row, "Category": rule_category, "Slot": slot_num,
                            "Image_Type": slot_info["image_type"]},
                "slot_num": slot_num,
                "slot_info": slot_info,
                "covered_url": covered.get(slot_num),
                "in_missing": slot_num in missing_slot_nums,
                "reference_url": reference_url,
                "rule_category": rule_category,
                "product_id": str(row["Product_ID"]),
                "sku": row["SKU"],
                "product_name": row["Name"],
                "description": row.get("Description", ""),
                "specifications": row.get("Specifications", ""),
                "forced_variation": slot_variations.get(slot_num, ""),
                "image_out_dir": image_out_dir,
            })
    return tasks


def process_slot_task(task: dict, project_id: str, region: str, tokens: VertexTokenProvider,
                      uploader: "GcsUploader | None", upload_cache: "UploadCache",
                      overwrite: bool) -> dict:
    """Does the real work for ONE (product, slot) task and returns a
    final_output row, tagged with an internal "_bucket"/"_needs_review" for
    the run-summary counts (stripped from what the reader sees — write_final_excel
    only looks up specific named columns, so these extra keys are harmless).

    Called from worker threads (see main()'s use of run_concurrent) —
    everything it touches (uploader, upload_cache, tokens) is already
    safe under concurrency: UploadCache/maybe_upload only ever mutate via
    a single dict-set + json.dump per call (fine — worst case is a
    redundant re-upload, not corruption), and VertexTokenProvider now
    locks its own refresh.
    """
    row_base = task["row_base"]
    if task["kind"] == "no_rule":
        return {**row_base, "Status": "no_rule_for_category", "_bucket": "Failed"}

    slot_num = task["slot_num"]
    slot_info = task["slot_info"]
    image_type_lower = slot_info["image_type"].lower()

    if task["covered_url"]:
        existing_filename = (f"{task['product_id']}_{slugify(task['sku'])}_{slot_num}_"
                             f"{slugify(slot_info['image_type'])}_existing_enhanced.png")
        check = ensure_existing_image_quality(
            task["covered_url"], existing_filename, task["image_out_dir"],
            uploader, upload_cache, 0)
        return {**row_base, "Status": check["status"], "Image_Source": check["image_source"],
               "GCP_Link": check["gcp_link"], "_bucket": "Existing",
               "_upload_failed": check["upload_failures"] > 0}

    if not task["in_missing"]:
        return {**row_base, "Status": "unknown_not_classified", "_bucket": "Failed"}

    if not task["reference_url"]:
        return {**row_base, "Status": "failed_no_reference_image", "_bucket": "Failed"}

    # Product_ID leads the filename (not just SKU) because that's the
    # identifier visible in the admin panel URL (?id=<Product_ID>) — the
    # one actually used to look a product up, so the filename/link alone
    # tells you the product AND which of the 6 slots it is without
    # opening anything else.
    filename = (f"{task['product_id']}_{slugify(task['sku'])}_{slot_num}_"
               f"{slugify(slot_info['image_type'])}.png")
    out_path = os.path.join(task["image_out_dir"], filename)

    if os.path.exists(out_path) and not overwrite:
        gcp_link, upload_failures = maybe_upload(uploader, upload_cache, filename, out_path, 0)
        return {**row_base, "Status": "Generated", "Image_Source": filename,
               "GCP_Link": gcp_link, "_bucket": "Reused",
               "_upload_failed": upload_failures > 0}

    prompt = build_generation_prompt(task["product_name"], task["rule_category"], slot_info,
                                     task["description"], task["specifications"],
                                     task["forced_variation"])
    # Dimension-type slots have a checkable ground truth (an exact, named
    # set of measurements) — verify the output actually shows them
    # correctly and retry (capped at MAX_GENERATION_ATTEMPTS) instead of
    # trusting one generation call got it right. Real output has
    # repeatedly shown a dropped/duplicated measurement or a stray arrow
    # even with an explicit prompt.
    axis_labels = []
    is_vehicle = False
    text_only_label = ""
    if "size" in image_type_lower or "dimension" in image_type_lower:
        axis_labels = compute_axis_labels(task["description"], task["specifications"])
        is_vehicle = is_vehicle_product(task["rule_category"], task["product_name"],
                                        task["description"], task["specifications"])
        # Keep verification in sync with what the prompt actually asked
        # for — vehicles get the side-profile 2-arrow layout (Breadth as
        # text only, no arrow), so the verifier must check for 2 arrows,
        # not 3, or it would flag a correct image as missing one. The
        # text-only label still needs its own spelling checked, so it's
        # passed through separately rather than discarded.
        axis_labels, text_only_label = resolve_dimension_layout(axis_labels, is_vehicle)
    result = generate_image_with_verification(
        task["reference_url"], prompt, out_path, project_id, region, tokens, axis_labels,
        is_vehicle=is_vehicle, text_only_label=text_only_label)
    status = result["status"]
    generated = status == "generated" or status.startswith("generated_unverified")
    gcp_link = ""
    upload_failed = False
    if generated:
        gcp_link, upload_failures = maybe_upload(uploader, upload_cache, filename, out_path, 0)
        upload_failed = upload_failures > 0
    if status == "generated":
        display_status = "Generated"
    elif status.startswith("generated_unverified"):
        display_status = "Generated (needs review)"
    else:
        display_status = status
    return {**row_base, "Status": display_status,
           "Image_Source": filename if generated else "", "GCP_Link": gcp_link,
           "_bucket": "Generated" if generated else "Failed",
           "_needs_review": status.startswith("generated_unverified"),
           "_upload_failed": upload_failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--classification", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--image_out_dir", default="generated_images")
    ap.add_argument("--out", default="final_output.xlsx")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate images that already exist in --image_out_dir "
                         "(default is to reuse them and not pay for them twice)")
    ap.add_argument("--limit", type=int,
                    help="only process the first N products (for the 5-10 product "
                         "sanity check)")
    ap.add_argument("--workers", type=int, default=8,
                    help="max concurrent Vertex AI / GCS calls in flight at once "
                         "(default 8). This is also the effective rate-limit control — "
                         "raise it for a faster bulk run only as far as your Vertex AI "
                         "quota actually allows; every call already retries 429s with "
                         "backoff, but a --workers set far above your quota will just "
                         "mean most of them spend their time retrying instead of working.")
    args = ap.parse_args()

    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        sys.exit("ERROR: set GCP_PROJECT_ID environment variable first.")
    region = os.environ.get("GCP_REGION", "us-central1")
    tokens = VertexTokenProvider(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    uploader = build_gcs_uploader()

    os.makedirs(args.image_out_dir, exist_ok=True)
    upload_cache = UploadCache(os.path.join(args.image_out_dir, ".gcs_uploads.json"))
    rules = RuleMaster.load(args.rules)
    merged = load_inputs(args.products, args.classification)
    if args.limit:
        merged = merged.head(args.limit)

    tasks = build_slot_tasks(merged, rules, args.image_out_dir)
    total = len(tasks)

    # Every (product, slot) is checkpointed by key immediately after it
    # completes — a crash or Ctrl-C partway through a 1000+ product run
    # loses at most the handful of tasks that were in flight at that
    # instant when re-run with the same --out, not hours of already-paid
    # for API calls.
    checkpoint = Checkpoint(args.out + ".checkpoint.jsonl")
    already_done = sum(1 for t in tasks if checkpoint.is_done(t["key"]))
    if already_done:
        print(f"Resuming from checkpoint: {already_done}/{total} slot-tasks already done.")

    progress_lock = threading.Lock()
    # Starts at 0, not already_done: on_result fires once per task in
    # `tasks` regardless of whether the worker actually did fresh work or
    # just returned a cached checkpoint hit (see worker() below), so this
    # naturally counts up to `total` exactly once either way.
    progress = {"n": 0}
    live_counts = {"Existing": 0, "Generated": 0, "Reused": 0, "Failed": 0, "NeedsReview": 0}
    status_path = args.out + ".status.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def worker(task):
        if checkpoint.is_done(task["key"]):
            return checkpoint.get(task["key"])
        row = process_slot_task(task, project_id, region, tokens, uploader, upload_cache,
                                args.overwrite)
        checkpoint.record(task["key"], row)
        return row

    def on_result(_i, _task, row):
        with progress_lock:
            progress["n"] += 1
            n = progress["n"]
            live_counts[row.get("_bucket", "Failed")] += 1
            if row.get("_needs_review"):
                live_counts["NeedsReview"] += 1
        # Every task, not just every Nth — a 1-2k run is exactly where you
        # want to notice a stall quickly, and this is one print per slot,
        # not per product, so it's not excessive.
        if n % 10 == 0 or n == total:
            print(f"[{n}/{total}] done")
            # Best-effort — no one is watching a terminal on an unattended
            # VM run, so this is what a poller (cron, health check, or a
            # plain `cat` over SSH) checks instead. Piggybacks on the same
            # every-10 cadence as the print above rather than every single
            # task, to keep the extra file I/O negligible at 1-2k scale.
            write_status(status_path, total=total, done=n, started_at=started_at,
                        **live_counts)

    final_rows = run_concurrent(tasks, worker, max_workers=args.workers, on_result=on_result)
    checkpoint.close()
    write_status(status_path, total=total, done=total, started_at=started_at,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **live_counts)

    # on_result already tallied every task's bucket exactly once (fires for
    # cache-hit resumes too, not just fresh work — see worker() above), so
    # live_counts IS the final tally; only upload_failures needs a fresh
    # pass since on_result doesn't track it separately.
    counts = live_counts
    upload_failures = sum(1 for row in final_rows if row.get("_upload_failed"))

    write_final_excel(final_rows, args.out)
    print(f"\nDone -> {args.out}")
    print(f"Generated images saved in ./{args.image_out_dir}/")
    print(f"Slots: {counts['Existing']} existing, {counts['Generated']} newly generated, "
          f"{counts['Reused']} reused from disk, {counts['Failed']} failed/unresolved.")
    if counts["NeedsReview"]:
        print(f"{counts['NeedsReview']} dimension image(s) never passed verification after "
              f"{MAX_GENERATION_ATTEMPTS} attempts — marked \"Generated (needs review)\" in "
              f"the sheet, kept the last attempt rather than nothing.")
    if counts["Failed"]:
        print("Check the Status column in the sheet for the failure reasons.")
    if uploader:
        uploaded = sum(1 for r in final_rows
                      if r["Status"] in ("Generated", "Generated (needs review)",
                                        "Existing (enhanced)") and r["GCP_Link"])
        print(f"GCS uploads: {uploaded} link(s) filled in automatically"
              + (f", {upload_failures} failed (GCP_Link left blank for those — "
                 f"upload manually)." if upload_failures else "."))
    else:
        print("GCS auto-upload not configured (set GCS_BUCKET + GCS_CREDENTIALS) — "
              "upload Generated files from the image dir and fill GCP_Link by hand.")


def write_final_excel(rows: list, out_path: str):
    FONT_NAME = "Arial"
    HEADER_FILL = PatternFill("solid", fgColor="1F2937")
    HEADER_FONT = Font(name=FONT_NAME, size=10, bold=True, color="FFFFFF")
    BODY_FONT = Font(name=FONT_NAME, size=10)
    STATUS_COLORS = {"Existing": "D9F2D9", "Generated": "D6E4FF",
                     "Existing (enhanced)": "D6E4FF",
                     "Existing (quality_check_failed)": "FFF3CD",
                     "Generated (needs review)": "FFF3CD"}
    FAILED_FILL = "FFE0E0"
    THIN = Side(style="thin", color="D9D9D9")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Image_Status"
    ws.sheet_view.showGridLines = False

    headers = ["Product_ID", "SKU", "Name", "Category", "Slot", "Image_Type",
               "Status", "Image_Source", "GCP_Link (fill after upload)"]
    widths = [11, 16, 40, 20, 6, 24, 20, 46, 40]
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=i, value=h)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")
        c.border = BORDER
        ws.column_dimensions[get_column_letter(i)].width = widths[i - 1]
    ws.freeze_panes = "A2"

    for r_idx, row in enumerate(rows, start=2):
        plain_values = [row["Product_ID"], row["SKU"], row["Name"], row["Category"],
                        row["Slot"], row["Image_Type"], row["Status"]]
        for c_idx, val in enumerate(plain_values, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BORDER
        # Image_Source is a real URL for Existing rows (clickable), but just a
        # local filename for Generated rows (not a link — the file lives on
        # disk, not the web) — write_link_cell only makes it a hyperlink when
        # it actually starts with http(s).
        write_link_cell(ws, r_idx, 8, row["Image_Source"], url=row["Image_Source"],
                        font=BODY_FONT, border=BORDER)
        write_link_cell(ws, r_idx, 9, row.get("GCP_Link", ""), url=row.get("GCP_Link", ""),
                        font=BODY_FONT, border=BORDER)
        # Anything that isn't Existing/Generated is a problem row — make the
        # ones needing attention visible at a glance.
        fill_color = STATUS_COLORS.get(row["Status"], FAILED_FILL)
        ws.cell(row=r_idx, column=7).fill = PatternFill("solid", fgColor=fill_color)
        ws.row_dimensions[r_idx].height = 20

    wb.save(out_path)


if __name__ == "__main__":
    main()
