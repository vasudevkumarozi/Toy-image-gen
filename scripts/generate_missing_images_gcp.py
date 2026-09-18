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
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import openpyxl
import pandas as pd
import requests
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from pipeline_lib import (
    Checkpoint,
    DIMENSION_STATUS_AMBIGUOUS,
    DIMENSION_STATUS_MISSING,
    DIMENSION_STATUS_VERIFIED,
    GcsUploader,
    GcsUploadError,
    RuleMaster,
    VertexTokenProvider,
    classify_dimensions,
    extract_dimensions_from_description,
    is_battery_operated,
    get_image_bytes,
    parse_description_fields,
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

# check_square_and_min_px (a hard, no-API gate — every attempt, every slot)
# started rejecting real output at exactly 2472x2048 and 2048x2180 — the
# SAME non-square size on all 3 retry attempts, tied to that product's own
# reference photo's own aspect ratio, not a one-off glitch. Nothing in the
# prompt had ever explicitly told the model the output canvas itself must
# be square regardless of the reference photo's shape — this had likely
# been silently shipping non-square images before the square check existed
# (upscale_to_minimum only preserves aspect ratio, it can't fix this).
# Retrying alone can't fix a defect this deterministic; the prompt has to
# actually say it.
SQUARE_CANVAS_RULE = (
    "The output image file itself must be a PERFECT SQUARE (1:1 width-to-height "
    "ratio) — this is independent of the reference photo's own shape or aspect "
    "ratio, which may well be rectangular. Do not simply mirror the reference "
    "photo's proportions into the output canvas. Compose the scene (product, "
    "background, any arrows/labels) so it fills a square frame, adding more "
    "background/scene on whichever side is needed to reach a square canvas — "
    "never a non-square output, and never blank/empty padding bars to reach it."
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
    "fits, use a WIDER framing than the reference, never a tighter one."
)

# Split out from FRAMING_RULE so the ONE slot type that legitimately needs
# two instances of the product (the packed+unpacked combo shot, see
# is_packed_and_unpacked_combo_slot) can skip just this sentence without
# losing the rest of FRAMING_RULE's real crop-prevention rules — the two
# were previously one hardcoded string, which silently told the model "no
# duplicate view" on the one slot that was explicitly asked to render a
# second, unpacked instance right next to the packaged one.
ONE_VIEW_RULE = (
    "Show exactly one view; do not add a second overlapping or duplicate "
    "view of the product in the same image."
)

# A generated "rear/back angle" shot came back showing the doll's hair
# from behind AS IF still visible through a sealed blister-pack window,
# merged with printed box-back text/photos in the same frame — physically
# incoherent, since a sealed package can't show a rotated product. Used to
# offer a choice between that and the box's own back panel, but real runs
# kept picking the box anyway (a Hot Wheels car's "Second Angle" shot came
# back as the SAME box from a different angle, not the car) — a customer
# wants to actually SEE the product in an angle shot, not its box from yet
# another side. Simplified to a single mandatory outcome: no packaging at
# all, ever, in an angle-type shot.
PACKAGING_LOGIC_RULE = (
    "This shot must show the product ITSELF, fully removed from any box or "
    "blister pack, rotated to the requested view — no packaging of any kind "
    "anywhere in the frame. Never substitute the retail package (front, "
    "back, or any side of it) for this shot, even if the product is normally "
    "sold packaged — a customer looking at an angle/view shot wants to see "
    "the actual product, not its box."
)

# Requested specifically: packaging artwork/logo/text is exactly the kind
# of "invented_details" verify_generated_image_generic already checks for
# generically — this makes the instruction explicit for packaging shots
# specifically, rather than relying on the generic product-fidelity wording
# alone, since a subtly redrawn logo/font/color on a box is easy to miss
# next to "don't change the product."
PACKAGING_FIDELITY_RULE = (
    "This shot shows the product's retail packaging/box. The packaging's "
    "printed artwork, logo, brand name, color scheme, and any printed text "
    "must be reproduced EXACTLY as shown in the reference image — do not "
    "redesign, simplify, recolor, reword, or invent any part of the "
    "packaging's printed design. If any packaging text is too small or "
    "blurry to read clearly in the reference, reproduce it as faithfully as "
    "possible rather than substituting different wording or a cleaner-"
    "looking placeholder."
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
    "it names. The label text must name ONE real, specific feature or "
    "selling point of THIS exact product, taken only from the product "
    "facts given above (its description bullets or specification fields) "
    "— do not invent a feature that isn't mentioned there.\n"
    "Camera framing — this is the structural fix, not optional styling: "
    "frame this ENTIRE shot as a close-up crop on the area of the product "
    "that contains the specific feature you are highlighting, so that "
    "area fills at least 30-40% of the frame — the rest of the product may "
    "only be partially visible at the edges, softly out of focus, or "
    "cropped off. Do NOT shoot a distant or full-product view and then try "
    "to point a pointer/arrow at a small far-away spot from there — that "
    "approach has repeatedly produced wrong-part pointers on real images "
    "(a \"Free Wheel Mechanism\" label pointed at a mixer drum, then a "
    "window/pillar, then a bumper, then a headlight, all on the same "
    "vehicle, before the shot was reframed as a close-up on the wheel "
    "itself — after which it was correct). With the feature large and "
    "dominant in frame, add a short leader line from a small text label "
    "(a rounded pill/tag background) directly to a point INSIDE that "
    "same dominant feature — never on a part at the edge of the crop that "
    "merely happens to be nearby. Before placing the pointer, identify "
    "which specific visible part is physically responsible for the named "
    "feature, and point there specifically, not at a plausible-looking or "
    "conveniently close part. Position the label in open background "
    "space, never on top of the product or overlapping another label. "
    "Keep the label short (a few words), clearly legible, and set in a "
    "clean bold sans-serif font with good contrast against its background "
    "— polished and modern, like a real marketplace product-listing "
    "infographic, not handwritten.\n"
    "Self-check before finalizing: your label names a physical thing (e.g. "
    "a knob, wheel, handle, button, clasp, texture, or material) — does "
    "your pointer's tip actually land ON that distinct physical part, and "
    "not merely somewhere else in the same close-up crop (a printed "
    "illustration, a face, a decorative pattern, or any other part that "
    "just happens to be nearby)? A real image, cropped in correctly on a "
    "toy piece, still pointed \"Easy-Grip Wooden Knob\" at the printed "
    "zebra face instead of the actual round wooden knob/peg visible in "
    "the same photo — being close up was not enough by itself; the "
    "pointer still has to land on the specific part whose shape matches "
    "the label's words, not just anywhere within the crop."
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

# A real Iron Man lifestyle shot came back with the SAME figure shown twice
# in one scene — once loose in the child's hand, once still inside its box
# held by the parent — and that box's plastic window was rendered torn/
# ripped open. NO_OTHER_PRODUCTS_RULE didn't catch this because it bans a
# DIFFERENT product, not a second instance of THIS SAME one. Two separate
# defects worth calling out explicitly: (1) a duplicated product reads as
# "does this come as a set of two", and (2) a torn/ripped box always reads
# as damaged merchandise, regardless of whether it's an intentional
# "unboxing moment" — never acceptable in a marketing image.
NO_DUPLICATE_INSTANCE_RULE = (
    "Show exactly ONE physical instance of this product in the scene — do "
    "not also include a second copy of the SAME product elsewhere in "
    "frame, whether that second copy is fully boxed, partially boxed, or "
    "in the background. This applies even if it might seem like a natural "
    "'unboxing' moment — a customer must not see two of the same item and "
    "wonder if two are included. Separately: never depict this product's "
    "retail box torn, ripped, cut open, or damaged in any way — a torn or "
    "damaged box always reads as defective/damaged merchandise, which is "
    "not acceptable in a marketing image under any circumstance."
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

# A generated "Box Back Content" shot invented an entire back-of-box layout
# (bullet-point copy, mini character lineup, barcode) with no way to verify
# any of it against the real product, since the reference photo only shows
# the box FRONT — unlike a redrawn front logo (caught by
# PACKAGING_FIDELITY_RULE, which assumes the real design IS visible in the
# reference), there is no ground truth at all for a face that was never
# photographed. Inventing packaging text/layout that can't be checked is
# worse than just not showing that face.
BOX_BACK_CONTENT_RULE = (
    "This slot asks for the BOX'S BACK PANEL CONTENT specifically (bullet "
    "points, feature callouts, mini photos, or similar printed back-cover "
    "layout) — but the reference photo only shows this product's box from "
    "the front, so there is no real information about what the back panel "
    "actually looks like. Do NOT invent back-panel text, bullet copy, "
    "character artwork, or layout that isn't visible in the reference photo "
    "— fabricated packaging content cannot be verified and may not match "
    "the real product's actual box. Instead, show the box from a different, "
    "SAFELY INFERABLE angle that stays consistent with what the reference "
    "photo actually shows (e.g. a 3/4 angled view of the same front the "
    "reference shows, or a visible side panel) rather than inventing the "
    "one face you have no reference for."
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


def assign_slot_variations(slots: dict, missing_slot_nums: set) -> tuple:
    """Returns (assignments, feature_positions).

    assignments: {slot_num: concrete instruction} for slots that share a
    family with at least one other MISSING slot in this same product/run —
    see the rotation lists above. Slots in a family alone (nothing else
    missing with the same base name), or families we don't have a rotation
    for, get no entry — build_generation_prompt falls back to the slot's
    own rule text as before.

    feature_positions: {slot_num: (index, total)} for "feature" family
    slots only — this is separate from the generic rotation text above
    because the "feature" family gets a second, stronger fix layer (see
    process_slot_task / extract_distinct_features): the generic rotation
    category (e.g. "a specific included accessory") is just a topic
    nudge, not a guarantee, and a real product (a Barbie toy guitar) had
    feature_1 and feature_2 both land on "the tuning knobs" because the
    product simply has no distinct accessory for that category to land
    on. (index, total) lets process_slot_task assign each slot a concrete,
    product-specific, non-overlapping feature instead, when extraction
    succeeds.
    """
    families = {}
    for slot_num in sorted(missing_slot_nums):
        if slot_num not in slots:
            continue
        family = re.sub(r"\s*\d+$", "", slots[slot_num]["image_type"]).strip().lower()
        families.setdefault(family, []).append(slot_num)

    assignments = {}
    feature_positions = {}
    for family, slot_nums in families.items():
        rotation = SLOT_FAMILY_ROTATIONS.get(family)
        if not rotation or len(slot_nums) < 2:
            continue
        for idx, slot_num in enumerate(slot_nums):
            assignments[slot_num] = rotation[idx % len(rotation)]
        if family == "feature":
            for idx, slot_num in enumerate(slot_nums):
                feature_positions[slot_num] = (idx, len(slot_nums))
    return assignments, feature_positions


# Text-only Vertex AI model (same one already used for the dimension
# verify step) for extracting a ranked, product-specific feature list —
# NOT the image-generation model.
FEATURE_EXTRACT_MODEL = os.environ.get("GEMINI_CLASSIFY_MODEL", "gemini-2.5-flash")

FEATURE_EXTRACT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "features": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": (
                "Up to the requested count of DISTINCT physical features, "
                "parts, or selling points of this exact product, ranked "
                "most to least prominent/sellable. Each entry a short "
                "(2-6 word) concrete noun phrase naming a real, physically "
                "distinct part explicitly supported by the product text "
                "given — never invented, and never the same part listed "
                "twice under different wording."
            ),
        },
    },
    "required": ["features"],
}

# Generic filler words excluded when comparing two extracted feature phrases
# for overlap — everything else (including plain nouns like "handle" or
# "knob") is treated as significant, since a shared concrete noun is exactly
# what indicates two phrasings name the same physical part.
_FEATURE_DEDUPE_STOPWORDS = {
    "a", "an", "the", "of", "for", "with", "and", "or", "to", "in", "on",
    "is", "its", "this", "that", "these", "those", "at", "as", "by",
    "feature", "design", "part", "product", "easy", "premium", "quality",
    "various", "included", "interactive",
}


def _feature_content_words(phrase: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", phrase.lower())
           if w not in _FEATURE_DEDUPE_STOPWORDS}


def extract_distinct_features(product_name: str, category: str, description: str,
                              specifications: str, count: int, project_id: str,
                              region: str, tokens: "VertexTokenProvider") -> list:
    """Asks a text model to name up to `count` distinct, concrete physical
    features of this specific product, ranked by prominence, sourced only
    from its own description/specification text.

    feature_1, feature_2, feature_3 are each generated by a SEPARATE,
    independent image-generation call (see build_slot_tasks/
    process_slot_task) with no visibility into one another — the only
    thing telling them apart before this was a generic rotating category
    ("a button/switch", "an accessory", "a material/texture", "a
    compartment"). That's a topic nudge, not a guarantee: a real product
    (a Barbie toy guitar) had feature_1 and feature_2 both come back
    captioned "tuning knob(s)" because the guitar has no distinct
    accessory for the "accessory" category to land on, so the model fell
    back to re-describing the same obvious part. Extracting a concrete,
    ranked, product-specific list up front and assigning one entry per
    slot (with the others named as exclusions — see build_generation_prompt)
    removes that ambiguity instead of hoping a generic category avoids a
    collision.

    Returns [] (never raises) on any error, or if the product's own text
    doesn't support `count` genuinely distinct features — callers must
    fall back to the existing generic rotation in that case, exactly as
    before this function existed.
    """
    context = f"Product: {product_name} (category: {category})"
    if description:
        context += f"\nDescription: {description[:800]}"
    if specifications:
        context += f"\nSpecifications: {specifications[:800]}"
    prompt = (
        f"{context}\n\n"
        f"Read the ENTIRE description and specification text carefully before "
        f"answering — check every bullet point, material, mechanism, included "
        f"accessory/component, printed character or branding, safety feature, "
        f"and educational/skill callout for a genuinely separate physical "
        f"aspect, not just the first or most obvious one. A product with only "
        f"one standout feature mentioned repeatedly in different words (e.g. "
        f"an \"easy-grip handle\" described once as a comfort feature and "
        f"again as a motor-skill feature) has ONE feature, not two — do not "
        f"reword the same part to pad the list to {count}; return fewer "
        f"instead (see below).\n\n"
        f"List up to {count} DISTINCT physical features, parts, or selling "
        f"points of THIS specific product that would each make a good "
        f"individual close-up product photo — e.g. a specific control, "
        f"material, mechanism, included accessory, or design detail. Each "
        f"one must be a short (2-6 word) concrete noun phrase naming a "
        f"real, physically distinct part or detail explicitly supported by "
        f"the product text above — never invent one that isn't mentioned "
        f"or clearly implied, and never list the same part twice in "
        f"different words (e.g. do not list both \"tuning knobs\" and "
        f"\"tuning pegs\" as if they were two different features). Order "
        f"them from most to least prominent/sellable. If this product's "
        f"own text only actually supports fewer than {count} genuinely "
        f"distinct features, return fewer rather than padding the list "
        f"with a near-duplicate or a vague generic entry."
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{FEATURE_EXTRACT_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": FEATURE_EXTRACT_SCHEMA},
    }
    try:
        headers = {"Authorization": f"Bearer {tokens.token()}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            return []
        candidates = resp.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") or [] if candidates else []
        text = next((p["text"] for p in parts if "text" in p), None)
        if text is None:
            return []
        parsed = json.loads(text)
        features = [f.strip() for f in (parsed.get("features") or []) if f and f.strip()]
        # Defensive de-dupe in case the model still rephrases the same part
        # twice despite the instruction above — exact-string matching alone
        # isn't enough. A real product (a Peppa Pig cupcake toy) had its
        # extraction come back as "Easy-Grip Interactive Handle",
        # "Easy-Grip Handle", and "Easy-Grip Handle Joint" — three different
        # strings, so the old case-insensitive check let all three through,
        # and all three feature_N slots ended up highlighting the exact same
        # physical handle. Any shared significant word (e.g. "handle", or
        # "tuning" in the "tuning knobs"/"tuning pegs" example above) is
        # treated as the same physical part.
        seen_word_sets = []
        deduped = []
        for f in features:
            words = _feature_content_words(f)
            if any(words & prior for prior in seen_word_sets):
                continue
            seen_word_sets.append(words)
            deduped.append(f)
        return deduped
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError):
        return []


class FeatureExtractionCache:
    """product_id -> extract_distinct_features() result, computed once and
    shared across that product's feature_1/feature_2/feature_3 tasks.

    Each feature slot is processed as a separate thread-pool task (see
    process_slot_task), so without this cache each sibling would either
    pay for its own redundant extraction call, or worse, three independent
    calls could rank/word the same product's features differently,
    defeating the entire point of assigning them a single shared,
    non-overlapping list. Two threads racing a cache miss both calling
    extract_distinct_features is harmless (same class of tolerated race as
    UploadCache above) — worst case is one redundant call, never
    corruption, since both threads agree on whichever result lands first.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def get_or_extract(self, product_id: str, product_name: str, category: str,
                       description: str, specifications: str, count: int,
                       project_id: str, region: str, tokens: "VertexTokenProvider") -> list:
        with self._lock:
            cached = self._data.get(product_id)
        if cached is not None:
            return cached
        features = extract_distinct_features(product_name, category, description,
                                              specifications, count, project_id, region, tokens)
        with self._lock:
            self._data.setdefault(product_id, features)
            return self._data[product_id]


def build_generation_prompt(product_name: str, category: str, slot_info: dict,
                           description: str = "", specifications: str = "",
                           forced_variation: str = "", exclude_features: list = None,
                           no_feature_available: bool = False, layout_plan: dict = None,
                           geometry_analysis: dict = None, retry_feedback: str = "",
                           low_quality_reference: bool = False) -> str:
    dimension_note = ""
    image_type_lower = slot_info["image_type"].lower()
    if "size" in image_type_lower or "dimension" in image_type_lower:
        if layout_plan:
            # Explicit Measurement Layout Plan block — the actual GEOMETRY
            # decision (see build_measurement_layout_plan), stated as plain
            # labeled facts rather than leaving the image model to work out
            # orientation and axis mapping for itself on top of everything
            # else it already has to do (preserve the product, render the
            # scene, draw arrows, render text). This is additive — it does
            # NOT replace the detailed arrow-count/magnitude/shape rules
            # built below, which stay exactly as before.
            constraints_text = "\n".join(
                f"- {v}" for v in layout_plan.get("geometry_constraints", {}).values()
            ) or "- (no swap risk — see mapping above)"
            geometry_hint = ""
            if geometry_analysis and geometry_analysis.get("longest_physical_edge"):
                geometry_hint = (
                    f"\nReference-photo geometry (for orientation guidance only — the "
                    f"numeric mapping above is the actual source of truth): the longest "
                    f"physical edge visible on the reference product is "
                    f"{geometry_analysis['longest_physical_edge']!r}"
                )
                if geometry_analysis.get("second_edge"):
                    geometry_hint += f"; the second horizontal edge is {geometry_analysis['second_edge']!r}"
                if geometry_analysis.get("thickness_edge"):
                    geometry_hint += f"; the thickness edge is {geometry_analysis['thickness_edge']!r}"
                geometry_hint += "."
            dimension_note += (
                f"\n\nMEASUREMENT LAYOUT PLAN (the geometry decision for this image "
                f"— follow it exactly):\n"
                f"- Product type: {layout_plan['product_type']}\n"
                f"- Camera orientation: {layout_plan['orientation']}\n"
                f"- Length maps to: {layout_plan['length_axis']}\n"
                + (f"- Breadth/Width maps to: {layout_plan['breadth_axis']}\n"
                   if layout_plan.get('breadth_axis') else "")
                + f"- Height maps to: {layout_plan['height_axis']}\n"
                f"- Dimension mapping: {layout_plan['dimension_mapping']}\n"
                f"MANDATORY GEOMETRY CONSTRAINTS (computed from the real numbers — "
                f"do not violate these regardless of how the reference photo looks "
                f"from any particular angle):\n{constraints_text}{geometry_hint}\n"
                f"Do not distort, stretch, squash, or reshape the product to force "
                f"these proportions — reorient/recompose the shot instead; the "
                f"product's real shape must be preserved exactly."
            )
        if is_boxed_multipiece_product(category, product_name, description, specifications):
            # Requested: a peg puzzle's dimension shot showed the board
            # assembled/open with its wooden pegs placed in their slots —
            # accurate to the product, but reads as "scattered" rather
            # than a clean size reference. For this category, show the
            # CLOSED retail box instead.
            dimension_note += (
                "\n\nThis product is sold as a packaged retail box containing "
                "multiple loose pieces (puzzle pegs, blocks, or similar). For "
                "this Size/Dimensions image, depict the product in its "
                "CLOSED, SEALED retail packaging box — as it would look on a "
                "store shelf before opening — NOT the assembled or open "
                "product with pieces placed in slots, removed, or scattered "
                "around it. Draw the measurement arrows on that closed box."
            )
        elif is_typically_boxed_single_item(category, product_name, description, specifications):
            # A single Iron Man action figure's Size/Dimensions image
            # rendered inconsistently across runs — sometimes the closed
            # box, sometimes the bare figure lying flat with the SAME
            # numbers (20x10x5 cm) drawn on its own body, even though its
            # own listed height (9.5in/~24cm) doesn't match that bounding
            # box at all. This category is always sold sealed in box/
            # blister packaging, and the admin's printed dimensions are
            # therefore the PACKAGING's size, not the bare figure's — so
            # this must be forced the same way as the multipiece case
            # above, not left for the model to guess from the reference
            # photo alone.
            dimension_note += (
                "\n\nThis product is normally sold sealed in retail box or "
                "blister packaging, and the dimensions given above describe "
                "that PACKAGING, not the bare unpackaged figure/item on its "
                "own. For this Size/Dimensions image, depict the product in "
                "its CLOSED, SEALED retail packaging — as it would look on a "
                "store shelf before opening — NOT the bare figure/item "
                "removed from its box and laid on a surface. Draw the "
                "measurement arrows on that closed packaging, never on the "
                "bare item itself."
            )
        # Specifications' "Dimensions (LxBxH)" wins over anything embedded
        # in the free-text description on conflict — see
        # merge_product_fields for why these can genuinely disagree.
        real_dims = extract_dimensions_from_description(description, specifications)
        if real_dims:
            # Without this, the model invents plausible-looking but wrong
            # numbers on the measurement lines — it has no way to know the
            # real size just from a photo. Real data beats a nice-looking guess.
            dimension_note += (
                f"\n\nIMPORTANT — real measurements: the manufacturer lists this "
                f"product's actual size as {real_dims}. Use these exact numbers on "
                f"the measurement lines and labels. Do not invent different numbers."
            )
        else:
            dimension_note += (
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
                # A fixed LEFT/RIGHT positional convention used to replace
                # magnitude-matching here entirely (regardless of which
                # number was bigger) — but that directly CONTRADICTED
                # composition_note below, which unconditionally tells the
                # model "the arrow for a larger number must look longer",
                # and it also meant nothing ever verified the visual
                # proportion for a plain box (see verify_dimension_image,
                # which mirrored this same positional convention). A real
                # product (5344, Length re-paired to the bigger 28.5 cm
                # number — see _label_dimension_value) shipped with
                # "Length" correctly on its conventional side but visually
                # SHORTER than "Breadth" on screen, because the model had
                # two conflicting instructions and only one was ever
                # checked. Length/Breadth must now always be matched to
                # their edges by actual magnitude, consistently in both
                # the prompt and the verifier — no positional escape hatch.
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
                shape_note = ""
                height_label_full = next((l for l in axis_labels if l.startswith("Height")), None)
                if n == 3 and height_label_full and len(horizontal_labels) == 2:
                    # The arrow-proportionality instruction alone wasn't
                    # enough — a real image had the Height arrow measuring
                    # LONGER on screen (930px) than the Length arrow (130px)
                    # despite Length's real number (30) being 20x bigger
                    # than Height's (1.5). The labels were on the correct
                    # sides, but the box's actual rendered 3D SHAPE was
                    # wrong (tall/thick instead of thin/flat) — arrows drawn
                    # on top of a wrongly-proportioned box can't fix
                    # anything. This states the real proportions explicitly
                    # so the shape itself gets built correctly first.
                    h_val = axis_label_magnitude(height_label_full)
                    max_horizontal_val = max(axis_label_magnitude(l) for l in horizontal_labels)
                    if h_val > 0 and max_horizontal_val >= h_val * 2:
                        ratio = round(max_horizontal_val / h_val, 1)
                        shape_note = (
                            f" This product's real proportions: "
                            f"{', '.join(axis_labels)} — its Height is about "
                            f"{ratio}x SMALLER than its longest horizontal "
                            f"measurement, i.e. a THIN, FLAT box (like a slim "
                            f"board game or picture-frame box), not a tall or "
                            f"thick one. LIE IT DOWN: rest the product flat on "
                            f"its LARGEST face (the Length x Breadth face), the "
                            f"same way a book lies flat on a table — do NOT "
                            f"stand it upright on its thin edge (like a book "
                            f"standing on a shelf), even if the reference photo "
                            f"itself shows it standing upright; rotate it "
                            f"roughly 90 degrees from that standing pose so it "
                            f"lies flat instead. Once it is lying flat, the "
                            f"Height edge is naturally just a thin sliver "
                            f"rising slightly at the near corner — clearly the "
                            f"thinnest thing in the photo — while the two long "
                            f"Length/Breadth edges dominate the frame along the "
                            f"ground. A real image stood the box upright "
                            f"instead, which made the thin Height edge fill "
                            f"almost the entire frame height and shrank the "
                            f"actual long edges to almost nothing — that is "
                            f"wrong no matter how the arrows are labeled; get "
                            f"the box lying flat first, then draw arrows that "
                            f"match it."
                        )
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
                    f"any of the {n} listed measurements.{magnitude_note}"
                    f"{composition_note}{shape_note} "
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
        # The "Size may vary slightly" disclaimer is now added afterward by
        # a deterministic PIL post-processing step (add_size_disclaimer),
        # not by this model call — asking the model to render it itself
        # gave no guarantee of exact wording, font, or position, and this
        # way it can never be missing or duplicated across retries. Tell
        # the model explicitly not to attempt its own version so it doesn't
        # draw a second, conflicting disclaimer in the same bottom-right
        # corner the post-processing step will use.
        dimension_note += (
            "\n\nDo NOT add any \"size may vary\" disclaimer or similar "
            "caveat text yourself — that is added separately afterward. "
            "Leave the bottom-right corner of the image clear of any text, "
            "arrow, or label so that later addition has room."
        )
        # A real image left true blank white bars along the top and bottom
        # of the canvas — the product+arrows only filled a band in the
        # middle, not the whole square canvas. Distinct from FULL_BLEED_RULE
        # (that one's about scene photos with a floor/room); this is about a
        # plain product shot not being scaled to actually use the frame.
        dimension_note += (
            "\n\nFill the ENTIRE image canvas edge to edge — the product, its "
            "arrows, and its background must reach all four edges, with no "
            "blank white or empty bars/borders added at the top, bottom, or "
            "sides. Scale and position the whole composition (product + "
            "arrows + labels) to actually use the full frame — do not shrink "
            "it down into a smaller banded region in the middle of the "
            "canvas and leave the rest blank."
        )
        if retry_feedback:
            # Failure-specific retry (section 14 of the brief): the PREVIOUS
            # attempt's actual QC failure, not a generic "try again" — see
            # generate_image_with_verification's retry_prompt_fn.
            dimension_note += f"\n\n{retry_feedback}"
    # Every slot type: keep the whole product in frame — cropping at the
    # edges (e.g. a close-up "Feature" shot clipping the wheel) has shown up
    # in real output. Lifestyle specifically: no background blur/bokeh —
    # requested explicitly, and a blurred background reads as lower catalog
    # quality even though it's a common lifestyle-photography convention.
    # Computed early (not just where the rest of the packed+unpacked combo
    # logic lives further below) because ONE_VIEW_RULE must be skipped from
    # the very first rules assembled, not appended after the fact.
    is_packed_and_unpacked_combo_slot = ("unpacked and packed" in image_type_lower
                                        or "packed and unpacked" in image_type_lower)
    extra_rules = (FRAMING_RULE + " " + NO_INVENTED_ACCESSORIES_RULE + " "
                  + NO_FICTIONAL_EFFECTS_RULE + " " + SQUARE_CANVAS_RULE)
    if not is_packed_and_unpacked_combo_slot:
        extra_rules += " " + ONE_VIEW_RULE
    # Any slot that composes a real-world scene (not just the product on a
    # plain studio background) is where letterboxing showed up — plain
    # product shots on white are unaffected since "empty space around the
    # product" IS the correct white background there.
    if any(k in image_type_lower for k in ("lifestyle", "learning", "skills", "action")):
        extra_rules += " " + FULL_BLEED_RULE + " " + NO_DUPLICATE_INSTANCE_RULE
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
    if "feature" in image_type_lower and no_feature_available:
        # This product's own description/specification text doesn't support
        # as many distinct physical features as this category has Feature
        # slots (see extract_distinct_features / process_slot_task) —
        # forcing a caption+callout here anyway is exactly how a fake
        # "second feature" gets invented. Fall back to a plain extra product
        # view instead: no caption requirement, no close-up-crop requirement.
        extra_rules += (
            " This product does not have another distinct feature to call "
            "out here — do NOT invent one. Instead, show a clean, different "
            "full or three-quarter view of the complete product (no text "
            "label, no pointer/leader line, no close-up callout framing)."
        )
    elif "feature" in image_type_lower:
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
    # "Full Unpacked and Packed Front-Angle View" (Dolls & Doll House,
    # Action Figures & Collectibles) contains "angle" in its own name but
    # is the ONE slot type whose whole point is showing packaging TOGETHER
    # with the unpacked product — PACKAGING_LOGIC_RULE's blanket "no
    # packaging at all" would directly contradict this slot's own
    # requirement (a real defect: a doll's slot 1 came back as just the
    # bare doll, no box, exactly as if that rule had suppressed it).
    # (is_packed_and_unpacked_combo_slot computed earlier, for ONE_VIEW_RULE.)
    if (any(k in image_type_lower for k in ("angle", "second", "rear", "back", "opposite"))
            and not is_packed_and_unpacked_combo_slot):
        extra_rules += " " + PACKAGING_LOGIC_RULE
    if any(k in image_type_lower for k in ("box", "pack", "package")):
        extra_rules += " " + PACKAGING_FIDELITY_RULE
    if is_packed_and_unpacked_combo_slot:
        extra_rules += (
            " This shot must show TWO things together in the same frame: (1) the "
            "product's real matching retail package/box, and (2) the complete "
            "product fully removed from that packaging, placed next to it. The box "
            "must look like a genuine, intact retail package (its real printed "
            "artwork, logo, and colors) — never an empty, blank, or generic box, "
            "and never let the box hide or obstruct the unpacked product. Do not "
            "show only one of the two — both must be clearly visible.\n"
            "IMPORTANT — if the reference image already shows the product boxed/"
            "packaged: this is an ADDITION, not a replacement or transformation. "
            "Keep that same box exactly as shown in the reference, unchanged and "
            "still in frame, and ADD a second, unpacked instance of the same "
            "product next to it (as if a matching unit had been removed from an "
            "identical box) — do not remove, edit away, or transform the boxed "
            "product into its unpacked form; both the reference's boxed product "
            "AND a new unpacked one must both end up visible side by side."
        )
    if "back" in image_type_lower and ("content" in image_type_lower or "box" in image_type_lower):
        extra_rules += " " + BOX_BACK_CONTENT_RULE

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
    if low_quality_reference:
        # Set only via assess_reference_image (process_slot_task) — the
        # reference photo passed a PRODUCT-IDENTITY check (it's genuinely
        # this product) but was flagged as a poor photo in its own right
        # (blurry/low-res/badly cropped/badly lit). Telling the model this
        # explicitly, rather than letting it silently mirror the reference's
        # own poor quality into the output.
        context_note += (
            "\n\nNote on the reference photo: it is a genuine photo of this "
            "exact product, but the photo itself is low quality (blurry, "
            "low-resolution, badly cropped, or poorly lit). Use it only to "
            "understand what the product looks like — render a CLEAN, sharp, "
            "well-lit, high-resolution result. Do not carry over the "
            "reference photo's own blur, noise, or poor framing into the "
            "output."
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
        if exclude_features:
            # Without this, "show a different feature" is just a hope —
            # each feature_N slot is a SEPARATE API call with no visibility
            # into what its siblings actually produced, so nothing stops
            # two of them converging on the same obvious part anyway. A
            # real product (a toy guitar) had feature_1 and feature_2 BOTH
            # come back captioned "tuning knob(s)" for exactly this reason.
            # Naming the sibling slots' assigned features explicitly and
            # forbidding their reuse is the only thing that actually
            # prevents the collision, since the two calls can't otherwise
            # coordinate.
            caption_clause = (
                ", with its own different caption text," if not no_feature_available else ""
            )
            variation_note += (
                f" This product's OTHER Feature images already cover: "
                f"{'; '.join(exclude_features)}. Do NOT feature or emphasize "
                f"any of those same parts again in THIS image — "
                f"{forced_variation} must be visually and physically "
                f"distinct from every one of them{caption_clause} so no two "
                f"images of this product repeat each other."
            )

    # retry_feedback for a Size/Dimensions slot is already appended inside
    # dimension_note above — this covers every OTHER slot type (currently
    # just Lifestyle's scale-mismatch retries; see process_slot_task's
    # retry_prompt_fn), so a failure-specific correction isn't silently
    # dropped just because the slot isn't a dimension slot.
    if retry_feedback and not ("size" in image_type_lower or "dimension" in image_type_lower):
        variation_note += f"\n\n{retry_feedback}"

    # "No added text" is the right default everywhere else, but it directly
    # contradicts FEATURE_TEXT_LABEL_RULE above for Feature-type slots,
    # which requires exactly one short real caption.
    quality_note = ("high resolution, realistic photography, clean "
                    "professional catalog style, no watermark")
    if "feature" in image_type_lower and not no_feature_available:
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


SIZE_DISCLAIMER_TEXT = "Size may vary slightly"


def add_size_disclaimer(image_path: str) -> None:
    """Stamps the mandatory disclaimer onto a dimension image as a
    deterministic PIL post-processing step, run AFTER generation and QC
    verification have both already passed — not left to the generative
    model to render. A model asked to draw this text itself cannot
    guarantee exact wording (paraphrasing/typos), a fixed corner, or that
    it never collides with an arrow/label it also drew in the same call.
    Doing it here fixes the text, the font, and the position by
    construction instead of by another QC check. Overwrites image_path in
    place, preserving its original format; uses Pillow's built-in default
    font (ImageFont.load_default(size=...), Pillow >=10.1) rather than a
    named system font (e.g. Arial) so this renders identically on both this
    Mac and the GCP instance, with no dependency on OS-installed fonts.
    """
    img = Image.open(image_path)
    original_format = img.format or "PNG"
    base = img.convert("RGBA")
    width, height = base.size

    font_size = max(16, round(width * 0.022))
    font = ImageFont.load_default(size=font_size)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0, 0), SIZE_DISCLAIMER_TEXT, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = round(width * 0.02)
    pad = round(font_size * 0.35)
    x = width - margin - text_w
    y = height - margin - text_h
    # Bottom-right by construction, with padding kept inside the canvas —
    # this is what guarantees it stays in bounds and off the product/arrows
    # (which the generation prompt keeps out of that corner already),
    # rather than trying to detect empty space after the fact.
    draw.rectangle([x - pad, y - pad, x + text_w + pad, y + text_h + pad],
                  fill=(255, 255, 255, 160))
    draw.text((x - bbox[0], y - bbox[1]), SIZE_DISCLAIMER_TEXT, font=font,
              fill=(90, 90, 90, 255))

    stamped = Image.alpha_composite(base, overlay)
    if original_format in ("JPEG", "JPG"):
        stamped = stamped.convert("RGB")
    stamped.save(image_path, format=original_format)


def check_square_and_min_px(image_bytes: bytes, min_px: int) -> dict:
    """Deterministic, no-API-call hard gate: every catalog image must be an
    exact 1:1 square at least min_px on each side. upscale_to_minimum (above)
    already guarantees the size floor, but it scales both dimensions by the
    same factor, so it can never fix a non-square result on its own — a
    non-square output is a real generation defect. Cropping it square would
    risk cutting off product/text/arrows, so this is treated as a retryable
    failure (see generate_image_with_verification) rather than something to
    silently trim. Returns {"valid": bool, "reason": str}."""
    try:
        width, height = Image.open(BytesIO(image_bytes)).size
    except Exception as e:
        return {"valid": False, "reason": f"unreadable_image: {e}"}
    if width != height:
        return {"valid": False, "reason": f"non_square: {width}x{height}px (must be exactly 1:1)"}
    if width < min_px:
        return {"valid": False,
               "reason": f"below_resolution_floor: {width}x{height}px (minimum {min_px}px)"}
    return {"valid": True, "reason": ""}


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
    whatever the product has — the edit needs the real toy to preserve.

    That last fallback (image_urls[0], used only when `covered` is empty —
    i.e. classify_images.py rejected EVERY existing image for this product)
    has no validation of its own — it could be the exact image classify
    just rejected as wrong/unusable. process_slot_task now runs
    assess_reference_image()/ReferenceAssessmentCache unconditionally on
    whatever reference ends up chosen here (not just this fallback case) —
    trusting classify_images.py's own per-slot judgment alone let a
    fundamentally wrong product photo slip through undetected on a run
    where it happened to accept the photo for at least one slot.

    Kept as the single "primary"/identity-check reference — see
    build_reference_set for the newer per-slot reference selection used
    for actual generation.
    """
    if 1 in covered:
        return covered[1]
    if covered:
        return covered[min(covered)]
    return image_urls[0] if image_urls else ""


# Roles a product's own validated existing images (already vetted for
# product identity by classify_images.py — an image only ends up in
# `covered` if it accepted THAT specific photo as showing this product for
# some slot) get grouped into, keyed by scanning each covered slot's own
# image_type text for these keywords.
_PACKAGING_ROLE_KEYWORDS = ("box", "pack", "package")
_SECONDARY_ROLE_KEYWORDS = ("angle", "second", "opposite", "rear")


def build_reference_set(covered: dict, slots: dict, image_urls: list,
                        reference_quality: dict = None) -> dict:
    """Groups this product's own validated existing images by ROLE instead
    of collapsing everything down to the one single reference photo reused
    for every generation call regardless of what that slot actually needs
    to depict. A real defect (product 35806's "Box Back Content" slot)
    invented an entire back-panel design because the only reference it
    ever saw was whatever photo covered slot 1 (the front) — if the admin
    panel has a real photo covering a packaging-type slot, a "Box Back
    Content"/similar slot should see THAT photo instead, not the front view.

    reference_quality (from classify_images.py's Reference_Quality column —
    see build_classification_prompt/RESPONSE_SCHEMA) answers a DIFFERENT
    question than "is this slot covered": a photo can correctly satisfy its
    OWN slot while being poor raw material to generate a DIFFERENT slot
    from (too zoomed, product partly out of frame, packaging obscuring most
    of it). Candidates with usable_as_reference explicitly False are
    excluded from every role here — they still remain that slot's own
    deliverable (see covered_url/ensure_existing_image_quality, unaffected
    by this function) — and when more than one candidate qualifies for the
    same role, the one with the higher image_quality wins. Missing/absent
    quality data for a slot defaults to "usable" (True) so this is a pure
    additive improvement on classification_result.csv files predating this
    column, never a new way to regress to worse behavior.

    Returns {"primary": url|None, "secondary": url|None,
             "packaging": url|None, "feature": url|None,
             "all_valid": [urls...]}. A missing role is None — callers
    (see pick_slot_reference) must fall back to "primary", never treat a
    missing role as an error.
    """
    reference_quality = reference_quality or {}

    def _quality(slot_num):
        return reference_quality.get(str(slot_num), {})

    def _is_usable(slot_num):
        return _quality(slot_num).get("usable_as_reference", True)

    def _score(slot_num):
        return _quality(slot_num).get("image_quality", 0.5)

    usable_slots = {sn: url for sn, url in covered.items() if _is_usable(sn)}

    result = {"primary": None, "secondary": None, "packaging": None,
             "feature": None, "all_valid": list(covered.values())}
    role_candidates = {"packaging": [], "feature": [], "secondary": []}
    for slot_num, url in usable_slots.items():
        slot_info = slots.get(slot_num)
        image_type_lower = slot_info["image_type"].lower() if slot_info else ""
        if any(k in image_type_lower for k in _PACKAGING_ROLE_KEYWORDS):
            role_candidates["packaging"].append(slot_num)
        elif "feature" in image_type_lower:
            role_candidates["feature"].append(slot_num)
        elif any(k in image_type_lower for k in _SECONDARY_ROLE_KEYWORDS):
            role_candidates["secondary"].append(slot_num)
    for role, candidates in role_candidates.items():
        if candidates:
            best = max(candidates, key=_score)
            result[role] = usable_slots[best]

    if 1 in usable_slots:
        result["primary"] = usable_slots[1]
    elif usable_slots:
        result["primary"] = usable_slots[max(usable_slots, key=_score)]
    elif 1 in covered:
        # Nothing passed the usability bar, including slot 1 — fall back to
        # it anyway rather than leave primary empty when slot 1 IS covered;
        # still strictly better than pre-quality-data behavior, never worse.
        result["primary"] = covered[1]
    elif covered:
        result["primary"] = covered[min(covered)]
    elif image_urls:
        result["primary"] = image_urls[0]
    return result


def pick_slot_reference(reference_set: dict, slot_info: dict) -> str:
    """Chooses the best-suited reference image for ONE slot from the
    product's reference set — a "Box Back Content" slot gets the real
    packaging photo when the admin panel has one covering some packaging-
    type slot, a "Feature" slot gets a real feature-type photo, etc.,
    instead of every slot always reusing whatever covered slot 1. Falls
    back to "primary" (and, if that's also empty, "" — callers already
    treat an empty reference as failed_no_reference_image) whenever this
    product has no covered image of the specifically-needed role."""
    image_type_lower = slot_info["image_type"].lower()
    if any(k in image_type_lower for k in _PACKAGING_ROLE_KEYWORDS):
        candidate = reference_set.get("packaging")
    elif "feature" in image_type_lower:
        candidate = reference_set.get("feature")
    elif any(k in image_type_lower for k in _SECONDARY_ROLE_KEYWORDS):
        candidate = reference_set.get("secondary")
    else:
        candidate = None
    return candidate or reference_set.get("primary") or ""


REFERENCE_ASSESSMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "shows_correct_product": {
            "type": "BOOLEAN",
            "description": ("True if this image shows the product described below "
                           "(matching its name/brand/type), even if the photo "
                           "itself is blurry, low-resolution, badly cropped, or "
                           "poorly lit — judge PRODUCT IDENTITY only, not photo "
                           "quality. False only if it shows a genuinely different "
                           "product, or is too broken/unrelated to tell what it "
                           "shows at all."),
        },
        "defect_type": {
            "type": "STRING",
            "enum": ["none", "quality_only", "wrong_product_or_unusable"],
            "description": ("'quality_only' = right product, just a poor photo "
                           "(blur/low-res/bad crop/bad lighting) — still usable "
                           "as an edit reference, just needs a cleaner render. "
                           "'wrong_product_or_unusable' = cannot be trusted as a "
                           "reference for this product at all — do not generate "
                           "from it."),
        },
        "reason": {"type": "STRING"},
    },
    "required": ["shows_correct_product", "defect_type", "reason"],
}


def assess_reference_image(reference_url: str, product_name: str, description: str,
                           specifications: str, project_id: str, region: str,
                           tokens: VertexTokenProvider) -> dict:
    """Only called for the risky fallback path described in
    pick_reference_image() above — when classify_images.py rejected every
    existing image for a product, the old code silently used image_urls[0]
    with zero validation, potentially the exact image classify just
    rejected. This distinguishes two real cases before spending a
    generation call on it:
      - the right product, just a bad PHOTO (blurry/low-res/badly cropped)
        → still safe to use as a reference, just ask for a cleaner render.
      - the wrong product entirely, or genuinely unusable
        → must not be used as if it were a trustworthy reference at all.

    Fails open ({"defect_type": "none", ...}) on any error — this is an
    extra safety net on top of the pipeline, not something that should
    block a product's generation because of one flaky call.
    """
    try:
        content, media_type = get_image_bytes(reference_url)
    except (requests.RequestException, OSError) as e:
        return {"shows_correct_product": False, "defect_type": "wrong_product_or_unusable",
               "reason": f"could not download reference image: {e}"}
    prompt = (
        f"This image is supposed to be a photo of: \"{product_name}\".\n"
        f"Product facts: {description[:500]} {specifications[:500]}\n\n"
        f"Does this image actually show that product? Judge PRODUCT IDENTITY "
        f"only, not photo quality — a blurry, low-resolution, or badly cropped "
        f"photo of the CORRECT product is still usable and should be reported "
        f"as defect_type='quality_only', not 'wrong_product_or_unusable'."
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{DIMENSION_VERIFY_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": media_type, "data": base64.b64encode(content).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": REFERENCE_ASSESSMENT_SCHEMA},
    }
    try:
        headers = {"Authorization": f"Bearer {tokens.token()}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            return {"shows_correct_product": True, "defect_type": "none",
                    "reason": f"verification_error: HTTP {resp.status_code}"}
        candidates = resp.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") or [] if candidates else []
        text = next((p["text"] for p in parts if "text" in p), None)
        if text is None:
            return {"shows_correct_product": True, "defect_type": "none",
                    "reason": "verification_error: no_text_in_response"}
        return json.loads(text)
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        return {"shows_correct_product": True, "defect_type": "none",
                "reason": f"verification_error: {e}"}


class ReferenceAssessmentCache:
    """product_id -> assess_reference_image() result, computed once and
    shared across every slot task for that product — all of a product's
    slots share the exact same single reference image (see
    pick_reference_image), so this only needs checking once per product,
    not once per slot. Same thread-safety pattern as FeatureExtractionCache
    above."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def get_or_assess(self, product_id: str, reference_url: str, product_name: str,
                      description: str, specifications: str, project_id: str,
                      region: str, tokens: VertexTokenProvider) -> dict:
        with self._lock:
            cached = self._data.get(product_id)
        if cached is not None:
            return cached
        result = assess_reference_image(reference_url, product_name, description,
                                        specifications, project_id, region, tokens)
        with self._lock:
            self._data.setdefault(product_id, result)
            return self._data[product_id]


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
        # empirically — resolution stays low regardless), but harmless to
        # send in case a model that honors it becomes available later. The
        # MIN_OUTPUT_PX upscale below is what actually guarantees the floor.
        # imageConfig.aspectRatio, unlike imageSize, IS honored — confirmed
        # empirically: without it, this same model mirrored a non-square
        # reference photo's aspect ratio into the output (e.g. 1236x1024)
        # even though SQUARE_CANVAS_RULE explicitly told it not to in plain
        # text; the model just didn't reliably follow that instruction.
        # Setting this is what actually, deterministically forces a square
        # canvas — the prompt rule is kept as a second layer (e.g. it still
        # needs to know not to letterbox/pad within that square), but this
        # is the real fix, not the wording alone.
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                             "imageConfig": {"imageSize": "2K", "aspectRatio": "1:1"}},
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

MULTIPIECE_KEYWORDS = (
    # Puzzles / building sets — the original real example (a peg puzzle).
    "puzzle", "block set", "building block", "peg puzzle", "sound puzzle",
    "stacking", "knobbed", "cylinder block",
    # Pretend-play "set" toys where the natural photo is many small loose
    # pieces spread out — cooking utensils, makeup/vanity, tools, tea sets,
    # doctor kits, etc. — clarified as the actual intended scope: ANY
    # product whose components would otherwise be photographed scattered,
    # not puzzles specifically.
    "kitchen set", "cooking set", "cookware", "utensil", "tea set",
    "tableware", "dinner set", "makeup", "cosmetic", "vanity", "beauty set",
    "tool set", "tool kit", "toolkit", "doctor set", "doctor kit",
    "medical kit", "jewelry making", "craft kit", "art set", "art kit",
    "accessory set", "grooming set", "grocery set", "market set",
    "tea party",
)


def is_boxed_multipiece_product(category: str, product_name: str, description: str,
                                specifications: str) -> bool:
    """Whether the Size/Dimensions image should show the CLOSED retail
    packaging box rather than the assembled/open product. A real dimension
    image for a peg puzzle showed the board with its wooden pegs placed in
    their slots at an angle — accurate to how the product looks in use, but
    the user wants the closed box for this whole category of product
    instead (explicitly: not just puzzles — any product with many small
    loose components, like cooking-utensil or makeup sets), since an
    open/scattered multi-piece product reads as messy rather than a clean,
    unambiguous shelf-ready size reference. Real product data for this
    category rarely states an exact piece count in text, so this matches
    on keyword alone rather than requiring a "N pieces" pattern."""
    haystack = f"{category} {product_name} {description} {specifications}".lower()
    if any(k in haystack for k in MULTIPIECE_KEYWORDS):
        return True
    # General fallback beyond the keyword list: a "Contents" field listing
    # several distinct, comma-separated items (e.g. "1 Pan, 2 Plates, 3
    # Cups, 1 Spoon") is itself evidence of a multi-piece set, regardless
    # of what the product happens to be called.
    fields = {**parse_description_fields(description), **parse_description_fields(specifications)}
    contents = fields.get("contents", "")
    return contents.count(",") >= 2


# A real product (a 9.5-inch Hasbro Iron Man action figure, admin
# Dimensions 20x10x5 cm) had its Size/Dimensions image swing between runs:
# sometimes the closed retail box, sometimes the bare figure laid flat on
# the floor with the SAME numbers drawn onto its own body — a 9.5in
# (~24cm) standing figure obviously isn't a 20cm bounding box lying down,
# so those numbers are almost certainly the BOX's dimensions, not the
# figure's. is_boxed_multipiece_product doesn't catch this: a single
# action figure isn't "multiple loose pieces". Any category where the
# item is always sold sealed in box/blister packaging — and the printed
# admin dimensions are therefore packaging dimensions, not the bare item's
# — needs the same forced "closed box" treatment for exactly the same
# reason, independent of whether it's one piece or many.
ALWAYS_BOXED_CATEGORIES = ("action figures & collectibles",)


def is_typically_boxed_single_item(category: str, product_name: str, description: str,
                                   specifications: str) -> bool:
    haystack = _shape_haystack(category, product_name, description, specifications)
    return any(c in haystack for c in ALWAYS_BOXED_CATEGORIES)


def _keyword_in_haystack(keyword: str, haystack: str) -> bool:
    """Word-bounded match for a single short word (avoids "ball" matching
    inside "pinball"/"basketball", "jar" inside a longer compound, etc.) —
    a real board game whose Description said "Pinball-style marble shooter
    gameplay" was misclassified as cylindrical purely because plain
    substring matching found "ball" inside "Pinball". Multi-word phrases
    (e.g. "cars & rc") keep plain substring matching since word-boundary
    regex on a phrase containing "&" is unreliable and those are specific
    enough not to appear as a false substring of something else anyway."""
    if " " in keyword or "&" in keyword:
        return keyword in haystack
    return re.search(rf"\b{re.escape(keyword)}\b", haystack) is not None


def _shape_haystack(category: str, product_name: str, description: str,
                    specifications: str) -> str:
    """The text used to detect a product's PRIMARY physical shape (vehicle /
    cylindrical / flat) — deliberately EXCLUDES the "Contents" field, since
    that lists included accessories/components (e.g. "8 x Marbles Game
    Balls" as one piece of a board game), not the shape of the product
    itself. A real board game ("Dinosaur Shooting Arcade") was
    misclassified as "cylindrical" purely because its Contents list
    happened to mention "Balls" as an included accessory.

    Unlike is_boxed_multipiece_product, which WANTS to look at Contents (a
    comma-heavy Contents list is itself evidence of a multi-piece set),
    shape detection should not be swayed by what a product merely includes
    alongside itself.
    """
    fields = {**parse_description_fields(description), **parse_description_fields(specifications)}
    desc_without_contents, spec_without_contents = description or "", specifications or ""
    if fields.get("contents"):
        desc_without_contents = "\n".join(
            l for l in desc_without_contents.split("\n") if not l.strip().lower().startswith("contents"))
        spec_without_contents = "\n".join(
            l for l in spec_without_contents.split("\n") if not l.strip().lower().startswith("contents"))
    return f"{category} {product_name} {desc_without_contents} {spec_without_contents}".lower()


# "wheel" alone is a strong, deliberately broad vehicle signal (catches RC
# cars/trucks that don't otherwise say "cars & rc" or "RC toy") — but it
# also appears in non-vehicle product names like "Pottery Wheel" (a
# spinning craft tool). A real product ("Kriiddaank Pottery Wheel Peppa Pig
# ...") was misclassified as a vehicle and given the side-profile wheelbase
# treatment for its Size/Dimensions image purely because of that one word.
# Excluding this SPECIFIC observed phrase rather than removing the "wheel"
# keyword entirely, since real vehicle products genuinely rely on it.
NON_VEHICLE_WHEEL_PHRASES = ("pottery wheel",)


def is_vehicle_product(category: str, product_name: str, description: str,
                       specifications: str) -> bool:
    """Whether the wheel-landmark dimension rule applies — shared by
    build_generation_prompt and process_slot_task (which passes the result
    to generate_image_with_verification) so both use the exact same
    detection."""
    haystack = _shape_haystack(category, product_name, description, specifications)
    if any(p in haystack for p in NON_VEHICLE_WHEEL_PHRASES):
        return any(_keyword_in_haystack(k, haystack) for k in VEHICLE_KEYWORDS if k != "wheel")
    return any(_keyword_in_haystack(k, haystack) for k in VEHICLE_KEYWORDS)


CYLINDRICAL_KEYWORDS = ("ball", "globe", "drum", "cylinder", "bottle", "jar", "tube", "roller")


def is_cylindrical_product(category: str, product_name: str, description: str,
                           specifications: str) -> bool:
    """A cylindrical product (ball, globe, drum, bottle...) has no separate
    Breadth axis distinct from its Length/diameter — Length and Breadth are
    the same measurement (the diameter) viewed from different sides, so
    there's no swap risk to check between them, only diameter vs height."""
    haystack = _shape_haystack(category, product_name, description, specifications)
    return any(_keyword_in_haystack(k, haystack) for k in CYLINDRICAL_KEYWORDS)


def is_flat_product(category: str, product_name: str, description: str,
                    specifications: str) -> bool:
    """Reuses the same keyword list FLAT_PRINT_RULE already uses for the
    Feature slot (a flat printed item has no 3D parts to zoom into, and
    for Dimensions it should be shot flat-on rather than at the 3/4 corner
    angle used for boxes, since there's no second horizontal ground-plane
    edge to show)."""
    flat_print_keywords = (
        "book", "workbook", "notebook", "flash card", "flashcard",
        "sticker book", "colouring", "coloring", "puzzle book",
        "activity pad", "activity book",
    )
    haystack = _shape_haystack(category, product_name, description, specifications)
    return any(_keyword_in_haystack(k, haystack) for k in flat_print_keywords)


SOFT_FOLDABLE_KEYWORDS = (
    "cape", "costume", "cloak", "poncho", "blanket", "bib", "apron",
    "dress-up", "cotton", "fabric", "cloth", "polyester", "velvet", "felt",
    "plush", "fleece",
)


def is_soft_foldable_product(category: str, product_name: str, description: str,
                             specifications: str) -> bool:
    """A soft/foldable/wearable item's admin "Dimensions" is its PACKAGED/
    FOLDED size, not its worn/unfolded silhouette — a real fabric cape kit
    (packaged 22.8x28.5x3.8 cm) rendered a child actually wearing the
    unfolded cape, which correctly looked ~1.5-2x the packaged figure. The
    Lifestyle scale check (verify_generated_image_generic's known_length_cm)
    flagged that as a false-positive SCALE_MISMATCH before this existed,
    because a rigid product (the Hot Wheels case this check was built for)
    has no such legitimate folded-vs-worn gap. Reuses the same
    _shape_haystack/_keyword_in_haystack pattern as is_vehicle_product/
    is_cylindrical_product/is_flat_product rather than a new ad-hoc check."""
    haystack = _shape_haystack(category, product_name, description, specifications)
    return any(_keyword_in_haystack(k, haystack) for k in SOFT_FOLDABLE_KEYWORDS)


DIECAST_SCALE_BRANDS = ("hot wheels", "majorette", "matchbox")
# An explicit "1:NN" anywhere in the name/spec always wins — some die-cast
# lines (Actonn RMZ, some Majorette sets) are 1:24, 1:32, 1:35, 1:36, 1:37, or
# 1:40, not the 1:64 "basic car" mainline. Falling back to 1:64 ONLY for the
# three brands whose mainline product is reliably that scale (a well-known,
# consistent collector fact for Hot Wheels/Matchbox/Majorette basic cars) —
# never guessed for any other brand, since a wrong assumption here is worse
# than the existing MANUAL_REVIEW_REQUIRED fallback.
_SCALE_RATIO_RE = re.compile(r"\b1\s*[:/]\s*(\d{2,3})\b")
# A generic real passenger car's approximate length/width/height in cm —
# NOT a per-model lookup (no real per-car database available). This is the
# same assumption underlying the actual observed real-world fact that
# mainline 1:64 Hot Wheels/Matchbox/Majorette cars are consistently ~7 cm
# long regardless of which real car they model, since manufacturers target a
# fixed packaged toy size, not a strict scale-accurate one. Used only when
# admin's own dimension data is MISSING/AMBIGUOUS for a confirmed die-cast
# scale-model vehicle.
_REFERENCE_CAR_CM = (440.0, 175.0, 145.0)


def detect_diecast_scale_ratio(category: str, product_name: str, description: str,
                               specifications: str) -> "int | None":
    """Returns the scale-model denominator (e.g. 64 for "1:64") if this is a
    die-cast scale-model vehicle whose real-world size can be defensibly
    estimated from its scale ratio, else None (routes to the existing
    MANUAL_REVIEW_REQUIRED path, unchanged)."""
    haystack = _shape_haystack(category, product_name, description, specifications)
    match = _SCALE_RATIO_RE.search(haystack)
    if match:
        return int(match.group(1))
    if any(_keyword_in_haystack(b, haystack) for b in DIECAST_SCALE_BRANDS):
        return 64
    return None


def estimate_diecast_dimensions_cm(scale_ratio: int) -> dict:
    length, breadth, height = (round(v / scale_ratio, 1) for v in _REFERENCE_CAR_CM)
    return {"length": length, "breadth": breadth, "height": height}


REFERENCE_DIMENSION_CHECK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "has_labeled_dimensions": {
            "type": "BOOLEAN",
            "description": ("True ONLY if this photo already has printed measurement "
                           "numbers with units (cm/mm/inch) attached to arrows or lines "
                           "drawn on the product, forming a real size/dimension diagram — "
                           "not just a plain product photo, and not a generic size chart "
                           "for a whole size-range (e.g. S/M/L clothing chart) that isn't "
                           "specific to this exact item's own measurements."),
        },
        "reason": {"type": "STRING"},
    },
    "required": ["has_labeled_dimensions", "reason"],
}

REFERENCE_PACKAGING_CHECK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "shows_packaged_product": {
            "type": "BOOLEAN",
            "description": ("True only if this photo shows the product genuinely "
                           "inside or behind its real retail packaging (a box, "
                           "blister pack, or similar) — the packaging's own printed "
                           "artwork must be visible, not just a plain product photo "
                           "with no packaging at all."),
        },
        "reason": {"type": "STRING"},
    },
    "required": ["shows_packaged_product", "reason"],
}


def image_shows_packaged_product(image_url: str, project_id: str, region: str,
                                 tokens: VertexTokenProvider) -> bool:
    """Checks whether a given photo already shows the product genuinely boxed/
    packaged (real printed retail artwork visible), for the deterministic
    packed+unpacked composite (see compose_packed_and_unpacked_image) — this
    is the one slot type ("Full Unpacked and Packed Front-Angle View") where
    an image-EDIT call kept dropping the box entirely when asked to render a
    second, unpacked instance next to it (three different prompt strategies
    all failed live, on product 2161 — a real doll whose admin panel DOES
    have a genuine boxed photo). Compositing two REAL states (the admin's
    own boxed photo + a separately generated unpacked photo) side by side is
    fully deterministic instead of depending on one edit call's compositional
    ability. Fails open (False) on any error — caller falls through to the
    existing single-call generative attempt, unchanged."""
    try:
        content, media_type = get_image_bytes(image_url)
    except (requests.RequestException, OSError):
        return False
    prompt = (
        "Look at this product photo. Is the product shown genuinely inside or "
        "behind its real retail packaging (a box or blister pack with visible "
        "printed artwork), rather than just the bare product with no packaging?"
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{DIMENSION_VERIFY_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": media_type, "data": base64.b64encode(content).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": REFERENCE_PACKAGING_CHECK_SCHEMA},
    }
    try:
        headers = {"Authorization": f"Bearer {tokens.token()}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            return False
        candidates = resp.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") or [] if candidates else []
        text = next((p["text"] for p in parts if "text" in p), None)
        if text is None:
            return False
        return bool(json.loads(text).get("shows_packaged_product"))
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError):
        return False


def compose_packed_and_unpacked_image(boxed_bytes: bytes, unpacked_bytes: bytes, out_path: str) -> None:
    """Pastes the two REAL photos (the admin's own boxed photo, and a
    separately generated clean unpacked photo) side by side on one square
    white canvas — boxed on the left, unpacked on the right, both scaled to
    the same height with a comfortable margin between them. Deterministic
    (no generative model composes anything), so both instances are
    guaranteed present, unlike asking one edit call to render both."""
    boxed = Image.open(BytesIO(boxed_bytes)).convert("RGB")
    unpacked = Image.open(BytesIO(unpacked_bytes)).convert("RGB")
    canvas_size = max(MIN_OUTPUT_PX, boxed.height, unpacked.height)
    margin = canvas_size // 20
    panel_h = canvas_size - 2 * margin
    gap = canvas_size // 25

    def _fit(img, target_h):
        scale = target_h / img.height
        return img.resize((max(1, round(img.width * scale)), target_h), Image.LANCZOS)

    boxed_r = _fit(boxed, panel_h)
    unpacked_r = _fit(unpacked, panel_h)
    total_w = boxed_r.width + gap + unpacked_r.width
    canvas_w = max(canvas_size, total_w + 2 * margin)
    canvas = Image.new("RGB", (canvas_w, canvas_size), (255, 255, 255))
    y = margin
    x = (canvas_w - total_w) // 2
    canvas.paste(boxed_r, (x, y))
    canvas.paste(unpacked_r, (x + boxed_r.width + gap, y))
    # Output must stay square per SQUARE_CANVAS_RULE — pad width-wise onto a
    # square canvas rather than leaving a rectangular result if the two
    # panels together end up wider than tall.
    if canvas_w > canvas_size:
        square = Image.new("RGB", (canvas_w, canvas_w), (255, 255, 255))
        square.paste(canvas, (0, (canvas_w - canvas_size) // 2))
        canvas = square
    canvas.save(out_path)


def reference_already_shows_dimensions(reference_url: str, project_id: str, region: str,
                                       tokens: VertexTokenProvider) -> dict:
    """Checks whether the admin's OWN reference photo for the Size/Dimensions
    slot already IS a proper labeled dimension diagram (arrows + printed
    numbers) — if so, that real photo should be used as-is instead of
    generating one, since it's strictly more trustworthy than anything we'd
    render. Almost every admin reference photo for this slot is just a plain
    product photo (no such diagram), so this is expected to return False for
    the large majority of products — it exists for the real minority where
    the manufacturer's own catalog image already has one.

    Fails open (False — fall through to the existing generation path,
    unchanged) on any error, same policy as assess_reference_image."""
    try:
        content, media_type = get_image_bytes(reference_url)
    except (requests.RequestException, OSError) as e:
        return {"has_labeled_dimensions": False, "reason": f"could not download: {e}"}
    prompt = (
        "Look at this product photo. Does it already show printed measurement "
        "numbers (with units like cm/mm/inch) attached to arrows or lines "
        "drawn directly on the product, forming a real, specific size/"
        "dimension diagram for this exact item?"
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{DIMENSION_VERIFY_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": media_type, "data": base64.b64encode(content).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": REFERENCE_DIMENSION_CHECK_SCHEMA},
    }
    try:
        headers = {"Authorization": f"Bearer {tokens.token()}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            return {"has_labeled_dimensions": False, "reason": f"HTTP {resp.status_code}"}
        candidates = resp.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") or [] if candidates else []
        text = next((p["text"] for p in parts if "text" in p), None)
        if text is None:
            return {"has_labeled_dimensions": False, "reason": "no_text_in_response"}
        return json.loads(text)
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        return {"has_labeled_dimensions": False, "reason": f"verification_error: {e}"}


def build_measurement_layout_plan(dim_data: dict, category: str, product_name: str,
                                  description: str, specifications: str) -> dict:
    """The actual GEOMETRY decision, made deterministically and up front —
    separate from asking the image model to draw anything (see section 6
    of the brief this implements: don't ask one model call to preserve the
    product, work out the geometry, AND render arrows all at once).

    Only meaningful for dim_data["status"] == VERIFIED_DIMENSIONS — callers
    must route AMBIGUOUS_DIMENSIONS/MISSING_DIMENSIONS to manual review
    before ever calling this (see process_slot_task), never guess a plan
    for unverified numbers.

    Returns a plan dict: product_type, orientation, {length,breadth,
    height}_axis (a physical/landmark description, not a screen direction),
    dimension_mapping ({"length": "20 cm", ...}), and geometry_constraints
    (plain-English statements of which axis must be visually larger than
    which — computed from the ACTUAL numbers, never assumed from position:
    test case "L=15, B=20" must produce the opposite constraint from
    "L=20, B=15", and "L=20, B=20" must produce no swap constraint at all).
    """
    is_vehicle = is_vehicle_product(category, product_name, description, specifications)
    is_boxed_multi = (is_boxed_multipiece_product(category, product_name, description, specifications)
                      or is_typically_boxed_single_item(category, product_name, description, specifications))
    is_flat = is_flat_product(category, product_name, description, specifications)
    is_cyl = is_cylindrical_product(category, product_name, description, specifications)

    length = dim_data.get("length")
    breadth = dim_data.get("breadth")
    height = dim_data.get("height")
    unit = dim_data.get("unit", "")

    if is_vehicle:
        product_type = "vehicle"
        orientation = "side_profile"
        length_axis = "wheelbase — front wheel to back wheel, same side"
        breadth_axis = None  # not visible edge-on from a side profile — text-only, see resolve_dimension_layout
        height_axis = "ground to the vehicle's own top surface, vertical"
    elif is_cyl:
        product_type = "cylindrical"
        orientation = "three_quarter"
        length_axis = "diameter — the widest horizontal span across the round body"
        breadth_axis = None  # same measurement as length for a round product, not a separate axis
        height_axis = "vertical height of the body"
    elif is_flat:
        product_type = "flat"
        orientation = "flat_front_on"
        length_axis = "long edge of the flat item, seen face-on"
        breadth_axis = "short edge of the flat item, perpendicular to the long edge"
        height_axis = "thin edge-on thickness, barely visible face-on"
    else:
        product_type = "box"
        orientation = "three_quarter_corner"
        length_axis = "the longer of the two ground-plane edges radiating from one near corner"
        breadth_axis = "the shorter of the two ground-plane edges radiating from that same corner"
        height_axis = "the vertical edge rising from that same corner"

    def _fmt(v):
        return f"{v:g} {unit}".strip() if v is not None else None

    dimension_mapping = {"length": _fmt(length), "breadth": _fmt(breadth), "height": _fmt(height)}

    geometry_constraints = {}
    if length is not None and breadth is not None and breadth_axis:
        if length != breadth:
            bigger, smaller = ("length", "breadth") if length > breadth else ("breadth", "length")
            ratio = max(length, breadth) / min(length, breadth)
            # A bare "longer than" was true whether the real ratio was
            # 1.05x or 10x — a box that's actually long and narrow could
            # still legally satisfy that wording while being drawn nearly
            # square. Stating the actual computed ratio (never a hardcoded
            # guess) is what stops that: the model has to reproduce roughly
            # how MUCH longer, not just which side is longer.
            if ratio < 1.15:
                proportion_note = (
                    "only marginally longer — the two edges should look close "
                    "in length, almost square"
                )
            else:
                # A subtle ratio (roughly 1.15x-1.5x) is where this keeps
                # failing in practice — a real box at exactly this ratio
                # (28.5 vs 22.8 cm) came back looking almost perfectly
                # square on screen even after being told "must look longer".
                # A follow-up attempt added a literal "if X spans 100px,
                # Y must span 125px" pixel example — that backfired badly:
                # the model rendered "2.8 cm"/"2.5 cm" instead of the real
                # 28.5/22.8, i.e. it visibly confused the EXAMPLE numbers
                # (100, 125) with the actual measurement numbers in the same
                # prompt. Never put invented placeholder numbers in the same
                # block as the real dimension values again — state the
                # requirement in words only.
                proportion_note = (
                    f"noticeably longer — roughly {ratio:.1f}x the length of the "
                    f"{smaller}_axis, not just marginally longer. This is a "
                    f"SUBTLE ratio that is easy to under-render as looking "
                    f"nearly square — a real image got this wrong for exactly "
                    f"this reason. If you are unsure whether the difference "
                    f"reads clearly at a glance, make {bigger}_axis MORE "
                    f"visibly longer rather than less; a viewer must be able "
                    f"to tell which edge is longer without measuring, even if "
                    f"that means erring slightly past the exact ratio in the "
                    f"direction of being clearly rectangular. Do not change "
                    f"the actual printed numbers/labels to achieve this — only "
                    f"the drawn edge lengths, never the measurement text."
                )
            geometry_constraints["length_vs_breadth"] = (
                f"{bigger}_axis must be visually the LONGER of the two horizontal "
                f"axes ({dimension_mapping[bigger]} > {dimension_mapping[smaller]}), "
                f"and {proportion_note} — do not render the box looking more "
                f"square than these real proportions warrant. The {smaller}_axis "
                f"must not be drawn longer."
            )
        else:
            geometry_constraints["length_vs_breadth"] = (
                f"length_axis and breadth_axis are the same real size "
                f"({dimension_mapping['length']}) — do not force either edge to "
                f"look longer than the other just because they have different names."
            )
    if height is not None and breadth is not None and breadth_axis:
        if breadth != height:
            bigger = "breadth_axis" if breadth > height else "height_axis"
            geometry_constraints["breadth_vs_height"] = (
                f"{bigger} must be visually the larger of the two where the product's "
                f"real shape makes that apparent (thin/flat products in particular must "
                f"not be rendered as tall/thick)."
            )
    elif height is not None and length is not None and not breadth_axis:
        # Cylindrical/vehicle case: no separate breadth axis, but height still
        # has a real relationship to length worth stating (e.g. a squat wide
        # ball vs a tall narrow bottle).
        if length != height:
            bigger = "length_axis" if length > height else "height_axis"
            geometry_constraints["length_vs_height"] = (
                f"{bigger} must be visually the larger of the two."
            )

    return {
        "product_type": product_type,
        "orientation": orientation,
        "length_axis": length_axis,
        "breadth_axis": breadth_axis,
        "height_axis": height_axis,
        "dimension_mapping": dimension_mapping,
        "geometry_constraints": geometry_constraints,
        "is_boxed_multipiece": is_boxed_multi,
    }


GEOMETRY_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "longest_physical_edge": {
            "type": "STRING",
            "description": ("Describe, in physical/landmark terms (never screen "
                           "directions like 'left'/'right', since the camera angle "
                           "used for generation may differ from this reference "
                           "photo's angle) which visible edge of the product is its "
                           "longest straight edge — e.g. 'the bottom edge of the "
                           "box, running front to back' or 'the top rim of the "
                           "lid'. This is the candidate edge for LENGTH."),
        },
        "second_edge": {
            "type": "STRING",
            "description": ("Describe the product's second horizontal edge, "
                           "perpendicular to the longest one — the candidate edge "
                           "for BREADTH/WIDTH. Leave blank for a round/cylindrical "
                           "product with no separate second edge."),
        },
        "thickness_edge": {
            "type": "STRING",
            "description": "Describe the vertical/thickness edge — the candidate edge for HEIGHT.",
        },
        "notes": {"type": "STRING"},
    },
    "required": ["longest_physical_edge", "thickness_edge"],
}


def analyze_product_geometry(reference_bytes: bytes, reference_content_type: str,
                             layout_plan: dict, project_id: str, region: str,
                             tokens: VertexTokenProvider) -> dict:
    """Pre-generation planning signal (section 4 of the brief this
    implements) — looks at the REAL reference photo and describes, in
    physical/landmark terms, which edge is the product's longest, second,
    and thickness edge, so build_generation_prompt can hand the image model
    a concrete, product-specific starting point instead of a generic "3/4
    corner" instruction it has to work out fresh for every product.

    This is a PLANNING SIGNAL ONLY — it never overrides the verified
    numeric L/B/H from build_measurement_layout_plan, and it is never used
    as the pass/fail authority (that stays with verify_dimension_image's
    deterministic checks on the FINAL generated image). Fails open (empty
    dict) on any error — a flaky call here should degrade to the previous
    generic instructions, not block generation.
    """
    prompt = (
        f"Look at this product photo. It has been classified as product "
        f"type '{layout_plan.get('product_type')}'. Describe its geometry "
        f"using physical landmarks (corners, edges, panels, rims) — never "
        f"screen directions like 'left' or 'right', since a different "
        f"camera angle will be used later. Identify the longest straight "
        f"edge, the second horizontal edge perpendicular to it (if any), "
        f"and the thin vertical/thickness edge."
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{DIMENSION_VERIFY_MODEL}:generateContent"
    )
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": reference_content_type,
                                "data": base64.b64encode(reference_bytes).decode()}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": GEOMETRY_ANALYSIS_SCHEMA},
    }
    try:
        headers = {"Authorization": f"Bearer {tokens.token()}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            return {}
        candidates = resp.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") or [] if candidates else []
        text = next((p["text"] for p in parts if "text" in p), None)
        if text is None:
            return {}
        return json.loads(text)
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError):
        return {}


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
        # A DIFFERENT defect from the horizontal-only check above: a real
        # image got Length vs Breadth right relative to EACH OTHER, but
        # rendered the whole box standing upright with a huge vertical
        # Height arrow (the SMALLEST number, e.g. 3.8 cm) towering over
        # tiny Length/Breadth arrows at the bottom — a thin flat box drawn
        # as if it were a tall thick one. longest_horizontal_arrow_label
        # never catches this because it explicitly ignores Height.
        "overall_longest_arrow_label": {
            "type": "STRING",
            "description": ("Among ALL THREE measurement arrows — Length, Breadth/"
                           "Width, AND Height together — which ONE is drawn "
                           "visually LONGEST on the page overall? Just the name. "
                           "Judge by actual on-screen arrow length, not which "
                           "number is bigger, and not which one you were told to "
                           "expect."),
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
        "label_full_texts": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": ("The COMPLETE text of EVERY measurement label in the "
                           "image — name AND number AND unit together (e.g. "
                           "\"Height 1.5 cm\") — including any label that has no "
                           "arrow of its own, just plain text. Transcribe each one "
                           "EXACTLY as it is spelled/rendered in the image, "
                           "character for character, even if it looks misspelled "
                           "or the number looks wrong (e.g. a missing decimal "
                           "point) — do NOT auto-correct it to what you think it "
                           "was supposed to say. One entry per label."),
        },
        # Landmark-based, not magnitude-based — a text-correct label can
        # still sit on the physically WRONG edge even when
        # longest_horizontal_arrow_label comes back "correct", if the
        # model's own perception of "longest" was itself mistaken (garbage
        # in, garbage out on that check).
        # Describing each arrow's edge independently, in physical terms,
        # gives a second, differently-grounded signal: if both come back
        # describing the SAME physical edge, that's a real defect no
        # magnitude or position check would catch.
        "length_edge_landmark": {
            "type": "STRING",
            "description": ("Only when there are two horizontal measurement "
                           "arrows: describe, in PHYSICAL/landmark terms (e.g. "
                           "'the bottom-front edge of the box' or 'the edge "
                           "nearest the hinge'), which physical edge of the "
                           "product the LENGTH arrow runs along. Never use "
                           "screen directions like 'left'/'right'. Leave blank "
                           "if not applicable."),
        },
        "breadth_edge_landmark": {
            "type": "STRING",
            "description": ("Same as length_edge_landmark, but for whichever "
                           "arrow is labeled Breadth/Width. Must describe a "
                           "DIFFERENT physical edge than length_edge_landmark "
                           "— if you find yourself describing the same edge "
                           "for both, that itself is the defect to report "
                           "here, not something to reconcile by picking one "
                           "description. Leave blank if not applicable."),
        },
        "valid": {"type": "BOOLEAN"},
        "arrow_count": {"type": "INTEGER"},
        "reason": {"type": "STRING"},
    },
    # longest_horizontal_arrow_label/overall_longest_arrow_label are now
    # required (not just optional-with-a-fallback) — the code path that
    # consumes them treats a blank answer as a failure when one was
    # expected (see verify_dimension_image), but forcing the model to at
    # least attempt an answer every time, rather than silently omitting a
    # perceptually-hard field, is the first line of defense against the
    # exact gap that let product 27913 ship with Height and Breadth's
    # arrows swapped.
    "required": ["arrows_found", "valid", "arrow_count", "reason",
                "longest_horizontal_arrow_label", "overall_longest_arrow_label"],
}

# Matches the brief's requested enum (section 13) — populated deterministically
# in Python below, from the SAME checks that already set valid=False, so a
# retry can be told exactly what to fix (see generate_image_with_verification's
# retry_prompt_fn) instead of getting an identical prompt three times.
FAILURE_ARROW_COUNT = "ARROW_MISPLACED"
FAILURE_MISSING_OR_DUPLICATE = "WRONG_DIMENSION_LABEL"
FAILURE_TEXT_MISMATCH = "WRONG_DIMENSION_VALUE"
FAILURE_AXIS_SWAP = "LENGTH_BREADTH_AXIS_SWAP"
FAILURE_GEOMETRY_MISMATCH = "GEOMETRY_MISMATCH"
FAILURE_SCALE_MISMATCH = "SCALE_MISMATCH"
# Wrong/redesigned product or an invented part/accessory/color not backed by
# the reference photo or product facts — the single non-negotiable check
# (see verify_generated_image_generic's same_product/invented_details
# fields): a generated image failing THIS is never "close enough", unlike a
# cosmetic framing issue, so process_slot_task treats it (and
# FAILURE_SCALE_MISMATCH) as grounds to fall back to a plain alternate photo
# for a Lifestyle slot rather than ship a known-wrong scene.
FAILURE_PRODUCT_AUTHENTICITY = "PRODUCT_AUTHENTICITY"
# A Feature slot's rendered text label came back misspelled/garbled — a
# known image-model text-rendering failure mode, same class of defect as
# the dimension pipeline's label-spelling check, just for Feature captions
# instead of measurement numbers.
FAILURE_FEATURE_LABEL_SPELLING = "FEATURE_LABEL_SPELLING"
# A Feature slot's pointer/leader line lands on the wrong part (or no part
# at all) — a real, repeated failure mode (see FEATURE_TEXT_LABEL_RULE's
# comment history: a "Free Wheel Mechanism" label pointed at a mixer drum,
# then a window, then a bumper, then a headlight, across different real
# products) that nothing previously verified after the fact.
FAILURE_FEATURE_POINTER_WRONG = "FEATURE_POINTER_WRONG"
# An angle/view-type slot showed the retail box/packaging instead of (or
# alongside) the bare product — real defects: a Hot Wheels car's "Second
# Angle" shot came back as the same box from a different angle, and a
# doll's "Angle 1" shot showed the box-with-doll combo instead of the bare
# doll. See PACKAGING_LOGIC_RULE, tightened to a single mandatory outcome.
FAILURE_PACKAGING_IN_ANGLE_SHOT = "PACKAGING_IN_ANGLE_SHOT"
# The opposite problem, for the one slot type that explicitly requires
# BOTH the box and the unpacked product together ("Full Unpacked and
# Packed Front-Angle View") — a real defect showed only the bare product,
# no box at all, for a doll's slot 1.
FAILURE_MISSING_PACKED_OR_UNPACKED = "MISSING_PACKED_OR_UNPACKED"

# 1.5x oversized / 0.6x undersized. known_length_cm is only ever computed
# for a confirmed rigid, non-soft/foldable product (is_soft_foldable_product
# excludes everything else upstream in process_slot_task) — so the
# legitimate "worn/unfolded looks bigger than boxed" gap a soft good would
# have never reaches this check at all, and the band doesn't need to be
# widened to accommodate it. Tightened from an earlier 0.5x-2.0x (which let
# a product rendered at nearly DOUBLE its real size pass) down to this —
# still a real margin for the verifier's own visual-estimate imprecision,
# but tight enough to actually catch a "product looks bigger than it really
# is" defect like the Hot Wheels case this was built for (~7.5 cm shown as
# ~18-20 cm, ~2.5x) instead of only catching more extreme misses.
def is_scale_ratio_mismatch(ratio: float) -> bool:
    return ratio > 1.5 or ratio < 0.6


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
    # Full text (name + number + unit), not just the name — a real image
    # spelled "Height" correctly but rendered "1.5 cm" as "1 5 cm" (dropped
    # the decimal point), which a name-only check would have missed
    # entirely since "Height" was spelled fine.
    expected_full_labels = axis_labels + ([text_only_label] if text_only_label else [])

    horizontal_labels = [l for l in axis_labels if not l.startswith("Height")]
    expected_longest_name = None
    expect_length_on_wheels = False
    magnitude_check = ""
    # A fixed LEFT/RIGHT positional convention used to replace this
    # magnitude check entirely for a plain (non-vehicle) box — but
    # build_generation_prompt's composition_note unconditionally tells the
    # model "the arrow for a larger number must look longer", so skipping
    # the magnitude check here meant that instruction was never actually
    # verified for a box, only for a vehicle. A real product (5344) shipped
    # with the bigger-numbered "Length" positioned on its conventional side
    # but visually SHORTER on screen than "Breadth" — the one thing that
    # was checked (position) passed, the one thing that mattered
    # (magnitude) never got checked at all. Always check magnitude now, for
    # every product type with two horizontal labels.
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

    # A DIFFERENT defect from the horizontal-vs-horizontal check above: that
    # one explicitly ignores Height, so it never catches a thin flat box
    # rendered standing tall/thick — a real image got Length vs Breadth
    # right relative to each other but drew a huge vertical Height arrow
    # (the SMALLEST of the three numbers) towering over tiny Length/Breadth
    # arrows. Only worth checking when the real gap is large enough to be
    # unambiguous (matches the 2x threshold build_generation_prompt's
    # shape_note already uses for this same scenario) — avoids flagging a
    # subtle, hard-to-judge near-equal case.
    expected_overall_longest_name = None
    if len(axis_labels) >= 2:
        overall_ordered = sorted(axis_labels, key=axis_label_magnitude, reverse=True)
        biggest, smallest = overall_ordered[0], overall_ordered[-1]
        if axis_label_magnitude(smallest) > 0 and axis_label_magnitude(biggest) >= axis_label_magnitude(smallest) * 2:
            expected_overall_longest_name = biggest.split()[0]
            magnitude_check += (
                f" Separately, considering ALL THREE arrows together (Length, "
                f"Breadth/Width, AND Height), report which ONE is drawn "
                f"visually LONGEST overall in overall_longest_arrow_label — "
                f"judge this purely by actual on-screen arrow length, not by "
                f"which number is bigger."
            )

    text_only_note = (
        f" The image should also show \"{text_only_label}\" as a plain text "
        f"label with no arrow of its own — include its full text in "
        f"label_full_texts too."
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
        f"space; (4) fill in label_full_texts with the EXACT text of every "
        f"label — name, number, AND unit — transcribed character for "
        f"character even if it looks misspelled or the number looks wrong "
        f"(e.g. a missing decimal point) — do not silently auto-correct "
        f"anything when transcribing it. Set valid=true only if (1)-(3) "
        f"hold for every arrow in arrows_found.{magnitude_check}"
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
        failure_categories = []
        # Deterministic override: don't trust the model's own combined
        # judgment for the swap-check — compare its raw perceptual report
        # against the expected order in plain code. A real check reported
        # valid=true while the bigger number sat on the visually shorter
        # edge, because folding "perceive" + "apply this specific logic"
        # into one holistic verdict let the logic silently fail even when
        # the underlying perception (if asked for directly) would show it.
        # BUG FOUND (product 27913, a real 19x9x58cm guitar box rendered with
        # Height 58cm on the SHORTEST edge and Breadth 9cm on the LONGEST):
        # every branch below only acted when the model's report was non-empty
        # ("if reported and ..."/"if overall_reported and ...") — none of
        # these fields are in DIMENSION_VERIFY_SCHEMA's "required" list, so
        # the verifier could (and did) leave them blank, which silently
        # SKIPPED the exact swap-detection check that exists for this, and
        # the image passed on attempt 1 with a real, visible axis swap. A
        # missing answer where one was expected must now fail closed (be
        # treated as unverified) instead of being treated as "nothing to
        # check" — every branch below now handles the empty case explicitly.
        if expected_longest_name:
            wheel_reported = (parsed.get("wheel_direction_arrow_label") or "").strip()
            horiz_reported = (parsed.get("longest_horizontal_arrow_label") or "").strip()
            if is_vehicle and expect_length_on_wheels:
                # Name-based check: "Length" must be the one on the wheels,
                # by definition, regardless of magnitude — matches the
                # generation-side instruction (see build_generation_prompt).
                if not wheel_reported:
                    valid = False
                    failure_categories.append(FAILURE_AXIS_SWAP)
                    reason = (f"axis_swap_detected: verifier did not report which arrow runs "
                             f"along the wheels — treating a missing answer as unverified "
                             f"rather than silently passing. ({reason})")
                elif wheel_reported.split()[0].lower() != "length":
                    valid = False
                    failure_categories.append(FAILURE_AXIS_SWAP)
                    reason = (f"axis_swap_detected: model reported '{wheel_reported}' as "
                             f"running along the wheels (front-to-back), but 'Length' is "
                             f"defined as that measurement for a vehicle regardless of "
                             f"which number is bigger. ({reason})")
            elif is_vehicle:
                if not wheel_reported:
                    valid = False
                    failure_categories.append(FAILURE_AXIS_SWAP)
                    reason = (f"axis_swap_detected: verifier did not report which arrow runs "
                             f"along the wheels — treating a missing answer as unverified "
                             f"rather than silently passing. ({reason})")
                elif wheel_reported.split()[0].lower() != expected_longest_name.lower():
                    valid = False
                    failure_categories.append(FAILURE_AXIS_SWAP)
                    reason = (f"axis_swap_detected: model reported '{wheel_reported}' as "
                             f"running along the wheels, but '{expected_longest_name}' has "
                             f"the larger number and should align with the wheels. ({reason})")
            else:
                if not horiz_reported:
                    valid = False
                    failure_categories.append(FAILURE_AXIS_SWAP)
                    reason = (f"axis_swap_detected: verifier did not report which horizontal "
                             f"arrow is visually longest — treating a missing answer as "
                             f"unverified rather than silently passing. ({reason})")
                elif horiz_reported.split()[0].lower() != expected_longest_name.lower():
                    valid = False
                    failure_categories.append(FAILURE_AXIS_SWAP)
                    reason = (f"axis_swap_detected: model reported '{horiz_reported}' as the "
                             f"visually longest horizontal arrow, but '{expected_longest_name}' "
                             f"has the larger number and should be longest. ({reason})")
        if expected_overall_longest_name:
            overall_reported = (parsed.get("overall_longest_arrow_label") or "").strip()
            if not overall_reported:
                valid = False
                failure_categories.append(FAILURE_GEOMETRY_MISMATCH)
                reason = (f"geometry_mismatch: verifier did not report which arrow is overall "
                         f"longest even though the real numbers make this unambiguous "
                         f"('{expected_overall_longest_name}' should clearly be longest) — "
                         f"treating a missing answer as unverified rather than silently "
                         f"passing. ({reason})")
            elif overall_reported.split()[0].lower() != expected_overall_longest_name.lower():
                valid = False
                failure_categories.append(FAILURE_GEOMETRY_MISMATCH)
                reason = (f"geometry_mismatch: model reported '{overall_reported}' as the "
                         f"overall longest arrow (Length/Breadth/Height combined), but "
                         f"'{expected_overall_longest_name}' has the largest number and "
                         f"should be the longest of all three — likely a thin/flat "
                         f"product rendered standing tall/thick instead. ({reason})")
        # Second, independently-grounded geometry check — a text-correct
        # label can still sit on the physically wrong edge even when the
        # magnitude/position checks above both "pass", if the model's own
        # perception feeding THOSE checks was itself mistaken. Landmark
        # descriptions don't rely on "longest"/"left" judgment at all; if
        # the model describes the same physical edge for both Length and
        # Breadth, that is a real, independently-detected defect.
        length_landmark = (parsed.get("length_edge_landmark") or "").strip().lower()
        breadth_landmark = (parsed.get("breadth_edge_landmark") or "").strip().lower()
        if length_landmark and breadth_landmark and length_landmark == breadth_landmark:
            valid = False
            failure_categories.append(FAILURE_GEOMETRY_MISMATCH)
            reason = (f"geometry_mismatch: model described the SAME physical edge "
                     f"('{length_landmark}') for both the Length and Breadth arrows — "
                     f"they must be two different edges. ({reason})")
        # Deterministic text check — real output rendered "Breadeth" for
        # "Breadth", "Heglt" for "Height", and (separately) dropped the
        # decimal point in "1.5 cm" so it read as "1 5 cm" (i.e. 15 cm) —
        # all passed the verifier because it was never asked to check the
        # label text at all. Comparing the model's own (not-auto-corrected)
        # full-text transcription against the exact expected string in
        # code catches both the name AND the number, the same way the
        # swap-check above catches a bad holistic judgment. Whitespace is
        # stripped before comparing since the model may report a stray or
        # missing space that isn't the defect we care about here.
        def _normalize_label_text(s: str) -> str:
            return re.sub(r"\s+", "", (s or "")).lower()

        reported_full_lower = [
            _normalize_label_text(t) for t in (parsed.get("label_full_texts") or [])
        ]
        # A distinct defect from a spelling/number typo: the model draws
        # TWO arrows for the same name (e.g. two "Length" arrows) and drops
        # a different required name (e.g. no "Breadth" at all) — seen
        # repeatedly once the fixed left/right positional convention was
        # removed in favor of pure magnitude-matching (see composition_note/
        # magnitude_note above). FAILURE_TEXT_MISMATCH's "check spelling"
        # retry feedback doesn't address this at all, so it kept repeating
        # across all 3 attempts; this gets its own category and its own
        # targeted correction.
        reported_names = [t.split()[0].lower() for t in (parsed.get("label_full_texts") or []) if t.split()]
        name_counts = {}
        for rn in reported_names:
            name_counts[rn] = name_counts.get(rn, 0) + 1
        expected_names = {lbl.split()[0].lower() for lbl in expected_full_labels}
        missing_names = [n for n in expected_names if name_counts.get(n, 0) == 0]
        duplicated_names = [n for n, c in name_counts.items() if c > 1 and n in expected_names]
        if missing_names and duplicated_names:
            valid = False
            failure_categories.append(FAILURE_MISSING_OR_DUPLICATE)
            reason = (f"missing_or_duplicate_label: '{missing_names[0].title()}' is "
                     f"completely missing while '{duplicated_names[0].title()}' is "
                     f"duplicated across multiple arrows — each of Length/Breadth/"
                     f"Height must appear exactly once. ({reason})")
        for expected_full in expected_full_labels:
            if _normalize_label_text(expected_full) not in reported_full_lower:
                valid = False
                failure_categories.append(FAILURE_TEXT_MISMATCH)
                reason = (f"text_error: expected a label reading '{expected_full}' "
                         f"but it was not found among the transcribed labels "
                         f"{parsed.get('label_full_texts')!r} — likely misspelled, a "
                         f"corrupted number (e.g. a missing decimal point), or "
                         f"missing from the image. ({reason})")
        if not valid and not failure_categories:
            # The model's own holistic "valid" verdict fired but none of our
            # named deterministic checks did (e.g. a genuinely extra/missing
            # arrow, or an overlap the free-text reason describes) — still
            # worth a category so retries and audit logs aren't blank.
            failure_categories.append(FAILURE_ARROW_COUNT)
        return {"valid": valid, "reason": reason, "failure_categories": failure_categories}
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        return {"valid": True, "reason": f"verification_error: {e}", "failure_categories": []}


_CM_PER_UNIT = {
    "cm": 1.0, "cms": 1.0, "centimeter": 1.0, "centimeters": 1.0, "centimetre": 1.0,
    "mm": 0.1, "mms": 0.1, "millimeter": 0.1, "millimeters": 0.1, "millimetre": 0.1,
    "m": 100.0, "meter": 100.0, "meters": 100.0, "metre": 100.0,
    "in": 2.54, "inch": 2.54, "inches": 2.54, '"': 2.54,
    "ft": 30.48, "feet": 30.48, "foot": 30.48,
}


def _to_cm(value: "float | None", unit: str) -> "float | None":
    """Admin dimension data is overwhelmingly cm (see classify_dimensions'
    real examples), so an unrecognized/blank unit defaults to a 1.0 (cm)
    factor rather than discarding the value — same "trust real data,
    don't invent it, but don't throw away a usable number either"
    philosophy as the rest of this module."""
    if value is None:
        return None
    return value * _CM_PER_UNIT.get((unit or "cm").strip().lower(), 1.0)


_AMBIGUOUS_DIMENSION_NUMBER_RE = re.compile(r"[\d.]+")
_AMBIGUOUS_DIMENSION_UNIT_RE = re.compile(r"([a-zA-Z]+)\s*$")


def _ambiguous_longest_dimension_cm(raw_text: str) -> "float | None":
    """A Lifestyle scale check only needs the product's longest physical
    edge, in cm — unlike the Size/Dimensions slot's arrow chart, it does
    NOT need to know WHICH axis (Length vs Breadth vs Height) that number
    belongs to. classify_dimensions() correctly refuses to build a labeled
    arrow diagram from AMBIGUOUS data (unproven axis order), but real
    numbers with no proven order are still perfectly usable here: whichever
    of the 2-3 numbers is biggest IS the longest edge regardless of which
    word the admin wrote next to it. Restricting the scale check to only
    VERIFIED data (as before) meant most products with an unlabeled
    "Dimensions / Size: 20 x 15 x 2 cm" got NO lifestyle scale check at all.
    Returns None if raw_text has no parseable numbers."""
    numbers = [float(n) for n in _AMBIGUOUS_DIMENSION_NUMBER_RE.findall(raw_text or "")]
    if not numbers:
        return None
    unit_match = _AMBIGUOUS_DIMENSION_UNIT_RE.search((raw_text or "").strip())
    unit = unit_match.group(1) if unit_match else ""
    return _to_cm(max(numbers), unit)


# Every slot EXCEPT Size/Dimensions used to ship after a single ungated
# generation call — verify_dimension_image only ever covered dimensions.
# This is the equivalent gate for everything else (feature, lifestyle,
# hero, packaging, angle, ...): compares the generated image against the
# ORIGINAL reference photo plus this product's own admin description/
# specification text, and catches a wrong/redesigned product, an invented
# part/color/accessory, or — for Feature slots — a caption naming a
# feature the product doesn't actually have or one a sibling Feature slot
# already covers.
GENERIC_VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "same_product": {
            "type": "BOOLEAN",
            "description": ("True only if the SECOND image (generated) shows the exact "
                           "same physical product as the FIRST image (reference) — same "
                           "shape, proportions, color, material, parts, and branding. "
                           "False if it looks like a different or redesigned product, a "
                           "different color variant, or has been visibly distorted."),
        },
        "invented_details": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": ("List any part, accessory, control, button, color, logo, or "
                           "design element visible in the generated image that is NOT "
                           "visible in the reference image and NOT mentioned in the "
                           "product facts text given below. Empty list if nothing is "
                           "invented."),
        },
        "feature_highlighted": {
            "type": "STRING",
            "description": ("Only if this image is supposed to be a Feature/close-up "
                           "callout shot: the ONE specific physical feature or part this "
                           "image highlights, in a few words. Leave blank for every other "
                           "slot type."),
        },
        "feature_supported_by_product_facts": {
            "type": "BOOLEAN",
            "description": ("Only relevant when feature_highlighted is filled in: true "
                           "only if that exact feature is explicitly supported by the "
                           "product facts text below — not invented, not assumed from the "
                           "category in general. Leave true if feature_highlighted is "
                           "blank."),
        },
        "feature_label_text_exact": {
            "type": "STRING",
            "description": ("Only if this image is a Feature/close-up callout shot with a "
                           "text label rendered on it: transcribe that label's text EXACTLY "
                           "as it visually appears, letter for letter — including any typo, "
                           "garbled character, or misspelling exactly as rendered, do not "
                           "auto-correct it. Blank if this is not a Feature slot or no text "
                           "label is visible on the image."),
        },
        "feature_label_spelled_correctly": {
            "type": "BOOLEAN",
            "description": ("Only relevant when feature_label_text_exact is non-blank: true "
                           "only if that exact rendered text is spelled correctly — real, "
                           "correctly-spelled words with no dropped/doubled/swapped letters, "
                           "no garbled or corrupted characters, and no gibberish. False for "
                           "ANY typo or corrupted character, even a single letter. Leave "
                           "true if feature_label_text_exact is blank."),
        },
        "feature_pointer_correct": {
            "type": "BOOLEAN",
            "description": ("Only relevant when feature_highlighted is filled in AND a "
                           "pointer/leader line is drawn from the text label to the "
                           "product: true only if that pointer/leader line's tip actually "
                           "touches or clearly indicates the SAME physical part named by "
                           "feature_highlighted — not a different part, not empty "
                           "background, not the edge of an unrelated component. False if "
                           "the pointer lands on the wrong part, on nothing, or if there "
                           "is no visible pointer/leader line connecting the label to the "
                           "product at all. Leave true if feature_highlighted is blank."),
        },
        "packaging_matches": {
            "type": "BOOLEAN",
            "description": ("Only relevant if EITHER image shows the product's retail "
                           "packaging/box: true only if the packaging's printed artwork, "
                           "logo, brand name, colors, and text in the generated image match "
                           "the reference image's packaging exactly — not redesigned, "
                           "recolored, reworded, or simplified. Leave true if no packaging "
                           "is shown in either image."),
        },
        "scale_reference_present": {
            "type": "BOOLEAN",
            "description": ("Only relevant for a Lifestyle-type image: true only if a "
                           "person, a person's hand, or another object of well-known "
                           "standard real-world size is clearly visible in the SECOND "
                           "(generated) image, usable to judge the product's real scale. "
                           "False if the product is shown alone with nothing to judge "
                           "scale against."),
        },
        "product_estimated_length_cm": {
            "type": "NUMBER",
            "description": ("Only meaningful if scale_reference_present is true: your "
                           "best visual estimate, in centimeters, of the product's "
                           "longest visible dimension in the SECOND image — judged by "
                           "comparing it proportionally to the reference person/object's "
                           "well-known real-world size (e.g. an adult hand is roughly "
                           "18 cm long, an adult is roughly 165-180 cm tall). 0 if "
                           "scale_reference_present is false."),
        },
        "scale_reason": {
            "type": "STRING",
            "description": ("Only relevant if scale_reference_present is true: name the "
                           "specific anchor used (e.g. \"child's hand\", \"the seated "
                           "child's height\") and the real-world size assumed for it."),
        },
        "duplicate_product_instance": {
            "type": "BOOLEAN",
            "description": ("True only if a SECOND physical instance of this SAME "
                           "product is visible anywhere else in the image — e.g. two "
                           "copies of the retail box, or the product shown loose AND "
                           "also visible boxed elsewhere in the same shot. A real "
                           "product ships in ONE unit; this applies to every slot type, "
                           "not just real-world scene shots (a real defect showed the "
                           "same box twice in a plain Components/Contents shot). False "
                           "if only one instance of the product appears anywhere in the "
                           "image."),
        },
        "box_shown_damaged": {
            "type": "BOOLEAN",
            "description": ("True if the product's retail box/packaging is visible "
                           "anywhere in the SECOND image and appears torn, ripped, cut "
                           "open, crushed, or otherwise damaged. False if no box is shown, "
                           "or the box shown is fully intact."),
        },
        "packaging_shown_in_angle_shot": {
            "type": "BOOLEAN",
            "description": ("Only relevant if this slot is meant to be a bare-product "
                           "angle/view shot (not a packaging slot): true if the "
                           "product's retail box, blister pack, or any packaging is "
                           "visible anywhere in the image instead of (or alongside) "
                           "the bare product. False if the bare, unpackaged product is "
                           "shown, and false if this is not an angle-type slot."),
        },
        "shows_both_packed_and_unpacked": {
            "type": "BOOLEAN",
            "description": ("Only relevant if this slot's type explicitly calls for "
                           "showing BOTH the retail package AND the unpacked product "
                           "together (a 'packed and unpacked' combo shot): true only "
                           "if the SECOND image clearly shows (1) a real, intact, "
                           "non-empty retail box/package with genuine printed artwork, "
                           "AND (2) the complete product removed from that packaging, "
                           "both visible in the same frame. False if either is "
                           "missing, if the box looks empty/blank/generic, or if the "
                           "box hides the unpacked product. Leave true if this slot "
                           "is not a packed-and-unpacked combo shot."),
        },
        "valid": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["same_product", "invented_details", "valid", "reason"],
}


def verify_generated_image_generic(image_bytes: bytes, reference_bytes: bytes,
                                   reference_content_type: str, product_name: str,
                                   description: str, specifications: str, image_type: str,
                                   forced_variation: str, exclude_features: "list | None",
                                   project_id: str, region: str,
                                   tokens: VertexTokenProvider,
                                   known_length_cm: float = None) -> dict:
    """Product-fidelity check for every non-dimension slot. Fails OPEN
    (valid=True) on any error calling the verifier itself, same policy as
    verify_dimension_image — a flaky verification call should not burn the
    retry budget or block the row.

    known_length_cm, when given, is the product's real longest physical
    dimension (from classify_dimensions' VERIFIED admin data, never
    guessed) — only passed for Lifestyle-type slots. A real Hot Wheels car
    (~7.5 cm long) came back rendered at what visually reads as ~18-20 cm
    in a lifestyle scene: the prompt already asked the model to keep scale
    realistic (see build_generation_prompt's "Scale check" note), but nothing
    ever verified it actually did, so a wrong scale shipped silently. This
    adds a real check on top of that prompt instruction, same as
    SQUARE_CANVAS_RULE needed imageConfig.aspectRatio behind it rather than
    trusting the wording alone.
    """
    exclude_note = ""
    if exclude_features:
        exclude_note = (
            f"\n\nThis product's OTHER Feature images are already assigned to cover: "
            f"{'; '.join(exclude_features)}. If feature_highlighted matches or clearly "
            f"overlaps with any of those, report feature_supported_by_product_facts as "
            f"false and explain the overlap in reason — a repeated feature is exactly the "
            f"defect this check exists to catch."
        )
    variation_note = f"\n\nThis image was specifically requested to show: {forced_variation}." if forced_variation else ""
    # Always checked, not just scene-type slots — a real defect showed the
    # SAME retail box twice in a plain "Components" shot, not a lifestyle
    # scene, so gating this to scene slots only missed it entirely.
    # "Full Unpacked and Packed Front-Angle View" is the ONE slot type
    # where a SECOND instance of the product is the actual requirement
    # (boxed + unpacked, side by side) — the generic duplicate check below
    # would otherwise directly contradict that.
    is_packed_and_unpacked_combo = ("unpacked and packed" in image_type.lower()
                                    or "packed and unpacked" in image_type.lower())
    duplicate_note = ""
    if not is_packed_and_unpacked_combo:
        duplicate_note = (
            "\n\nCheck carefully whether the SAME product physically appears more than "
            "once anywhere in this image (e.g. two copies of the box, or the product "
            "shown loose AND also visible boxed elsewhere in the same shot) — fill in "
            "duplicate_product_instance accordingly. A real product ships as ONE unit."
        )
    if not is_packed_and_unpacked_combo and any(
            k in image_type.lower() for k in ("lifestyle", "learning", "skills", "action")):
        duplicate_note += (
            " This is also a real-world scene shot: also check whether the "
            "product's box/packaging, if shown anywhere in the scene, looks torn, "
            "ripped, or damaged — fill in box_shown_damaged accordingly."
        )
    scale_note = ""
    if known_length_cm is not None:
        scale_note = (
            f"\n\nThis product's actual real-world longest dimension, AS "
            f"PACKAGED/FOLDED, is {known_length_cm:.1f} cm. Look for a person, "
            f"hand, or other familiar-sized object in the SECOND image and use "
            f"it to judge whether the product is rendered at a plausible "
            f"real-world scale — fill in scale_reference_present, "
            f"product_estimated_length_cm, and scale_reason accordingly. If the "
            f"product is a soft, foldable, or wearable item (e.g. a fabric cape, "
            f"costume, blanket, or bag), remember its UNFOLDED or WORN size is "
            f"expected to look noticeably bigger than this packaged figure — "
            f"that is normal, not a defect; only flag a mismatch if the size "
            f"still looks implausible even accounting for that. If nothing in "
            f"the scene can be used as a size reference, set "
            f"scale_reference_present to false."
        )
    # is_packed_and_unpacked_combo already computed above (duplicate_note) —
    # "Full Unpacked and Packed Front-Angle View" contains "angle" in its
    # own name but is the one slot type that REQUIRES packaging alongside
    # the product, so it must not be told "no packaging at all" (see the
    # matching generation-side exclusion in build_generation_prompt).
    angle_packaging_note = ""
    if (any(k in image_type.lower() for k in ("angle", "second", "rear", "back", "opposite"))
            and not is_packed_and_unpacked_combo):
        angle_packaging_note = (
            "\n\nThis slot must show the BARE product, not its retail box or "
            "packaging — check whether any box, blister pack, or packaging is "
            "visible anywhere in the SECOND image (instead of, or alongside, the "
            "bare product) and fill in packaging_shown_in_angle_shot accordingly."
        )
    if is_packed_and_unpacked_combo:
        angle_packaging_note = (
            "\n\nThis slot must show BOTH the real retail package AND the unpacked "
            "product together — check whether the SECOND image actually shows a "
            "genuine, non-empty, intact box with real printed artwork AND the "
            "complete product removed from it, both clearly visible, and fill in "
            "shows_both_packed_and_unpacked accordingly."
        )
    prompt = (
        f"The FIRST image is the ORIGINAL reference photo of a real product called "
        f"\"{product_name}\". The SECOND image is a generated ecommerce photo that is "
        f"supposed to depict the exact same physical product, edited only for scene, "
        f"background, or camera angle — never redesigned, recolored, or given parts it "
        f"doesn't actually have.\n\n"
        f"Product facts (the only source of truth for what this product actually has — "
        f"do not accept a feature or part that isn't backed by this text or clearly "
        f"visible in the reference image):\n{description[:800]}\n"
        f"{('Specification section: ' + specifications[:800]) if specifications else ''}\n\n"
        f"This image's slot type is: {image_type}.{variation_note}{exclude_note}{scale_note}"
        f"{duplicate_note}{angle_packaging_note}\n\n"
        f"Compare the two images carefully and fill in every field. Set valid=true only "
        f"if same_product is true, invented_details is empty, and (when "
        f"feature_highlighted is filled in) feature_supported_by_product_facts is true, "
        f"feature_pointer_correct is true, and (when a text label is rendered on the "
        f"image) feature_label_spelled_correctly is true."
    )
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/publishers/google/models/{DIMENSION_VERIFY_MODEL}:generateContent"
    )
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": reference_content_type,
                                "data": base64.b64encode(reference_bytes).decode()}},
                {"inlineData": {"mimeType": "image/png",
                                "data": base64.b64encode(image_bytes).decode()}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": GENERIC_VERIFY_SCHEMA},
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
        failure_categories = []
        # Deterministic overrides — same pattern as verify_dimension_image:
        # check the raw perceptual fields ourselves rather than fully
        # trusting the model's own combined "valid" verdict.
        if not parsed.get("same_product", True):
            valid = False
            reason = f"wrong_or_altered_product: {reason}"
            failure_categories.append(FAILURE_PRODUCT_AUTHENTICITY)
        invented = parsed.get("invented_details") or []
        if invented:
            valid = False
            reason = f"invented_details_detected {invented}: {reason}"
            failure_categories.append(FAILURE_PRODUCT_AUTHENTICITY)
        feature_highlighted = (parsed.get("feature_highlighted") or "").strip()
        if feature_highlighted and not parsed.get("feature_supported_by_product_facts", True):
            valid = False
            reason = f"unsupported_or_duplicate_feature '{feature_highlighted}': {reason}"
        feature_label_text = (parsed.get("feature_label_text_exact") or "").strip()
        if feature_label_text and not parsed.get("feature_label_spelled_correctly", True):
            valid = False
            reason = f"feature_label_misspelled '{feature_label_text}': {reason}"
            failure_categories.append(FAILURE_FEATURE_LABEL_SPELLING)
        if feature_highlighted and not parsed.get("feature_pointer_correct", True):
            valid = False
            reason = f"feature_pointer_wrong_part '{feature_highlighted}': {reason}"
            failure_categories.append(FAILURE_FEATURE_POINTER_WRONG)
        if not parsed.get("packaging_matches", True):
            valid = False
            reason = f"packaging_mismatch: {reason}"
        if parsed.get("duplicate_product_instance"):
            valid = False
            reason = f"duplicate_product_instance_in_scene: {reason}"
        if parsed.get("box_shown_damaged"):
            valid = False
            reason = f"box_shown_damaged: {reason}"
        if parsed.get("packaging_shown_in_angle_shot"):
            valid = False
            reason = f"packaging_shown_in_angle_shot: {reason}"
            failure_categories.append(FAILURE_PACKAGING_IN_ANGLE_SHOT)
        if not parsed.get("shows_both_packed_and_unpacked", True):
            valid = False
            reason = f"missing_packed_or_unpacked: {reason}"
            failure_categories.append(FAILURE_MISSING_PACKED_OR_UNPACKED)
        if (known_length_cm and parsed.get("scale_reference_present")
                and parsed.get("product_estimated_length_cm")):
            estimated = parsed["product_estimated_length_cm"]
            ratio = estimated / known_length_cm
            if is_scale_ratio_mismatch(ratio):
                valid = False
                direction = "oversized" if ratio > 1 else "undersized"
                reason = (
                    f"scale_mismatch: product appears ~{estimated:.0f} cm long vs its "
                    f"actual {known_length_cm:.1f} cm ({direction}, ~{ratio:.1f}x) — "
                    f"judged against {parsed.get('scale_reason', 'a visible reference')}: {reason}"
                )
                failure_categories.append(FAILURE_SCALE_MISMATCH)
        return {"valid": valid, "reason": reason, "failure_categories": failure_categories}
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        return {"valid": True, "reason": f"verification_error: {e}"}


# ---------------------------------------------------------------------------
# Deterministic dimension pipeline: generate a CLEAN product image (no
# arrows/text at all), find its real corners with classical OpenCV contour
# detection, and draw the measurement arrows/labels with PIL — instead of
# asking the generative model to both preserve the product AND draw exact
# arrows/numbers on it in one pass, which is what produced every axis-swap
# and wrong-angle dimension defect this session (products 5344, 35806,
# 27913, ...).
#
# An earlier prototype tried the "find the corners" step by asking a vision
# LLM for pixel coordinates directly — that failed 2/2 live tests, landing
# points in empty background instead of on the product, because precise
# pixel localization is a known weak spot for these models. Classical CV
# (threshold/segment the plain background away, trace the actual contour)
# doesn't have that weakness — it's pixel math, not perception — confirmed
# on 2/2 real generated product images in live testing.
#
# This is intentionally NOT a wholesale replacement of the existing
# model-drawn-arrows + verify/retry system: every step here can fail
# (segmentation uncertain, corners don't resolve into 3 clean axes, etc.),
# and on ANY uncertainty this deliberately returns None/False so the caller
# falls back to the existing, already-proven path unchanged. Opportunistic
# improvement, zero regression risk.
# ---------------------------------------------------------------------------

CLEAN_PRODUCT_BACKGROUND_RULE = (
    "Show the product on a PERFECTLY FLAT, solid, plain white or very light "
    "gray background — no gradient, no vignette, no shadow darkening toward "
    "the edges or corners, no studio backdrop texture. This background must "
    "be uniform enough that a simple pixel-brightness threshold could "
    "separate the product from it automatically — an automated measurement "
    "step run after this image is generated depends on that. "
    "Do NOT draw, print, or overlay any arrows, lines, measurement labels, "
    "numbers, or any added text anywhere in this image — show ONLY the "
    "clean product itself. Fill the entire square canvas edge to edge."
)

# A real generated "clean" image (product 27731, a Hot Wheels car) came back
# technically correct — right product, flat background, no arrows — but
# rendered SMALL in the middle of the canvas with large empty margins on
# every side, like a normal breathing-room product shot. The automatic
# corner-detection step that runs on this image then only had a small
# cluster of pixels to work with, so the arrows/labels it computed came out
# tiny and cramped/overlapping in the final image — a defect in FRAME
# COMPOSITION, not in the detection or drawing logic. This is the opposite
# instruction from FRAMING_RULE's generous-margin guidance used elsewhere —
# deliberately, since precise pixel-level corner detection needs the product
# as LARGE as possible, not comfortably small.
CLEAN_PRODUCT_FILL_FRAME_RULE = (
    "The product must be the DOMINANT, LARGE subject of this image — zoom in "
    "as close as possible so it fills at least 80% of the canvas width or "
    "height (whichever is its longer visible dimension), with only a small "
    "margin of background visible around it. Do NOT render it small or "
    "distant with large empty white space on all sides like a normal "
    "breathing-room product photo — a small product in a mostly-empty frame "
    "gives an automated measurement step too few pixels to work with, "
    "producing cramped, unreadable measurement arrows and labels. The "
    "product must still be fully inside the frame, not cropped or touching "
    "the edge — maximize its size within that constraint, err on the side of "
    "too large/close rather than too small/distant."
)


def build_clean_dimension_prompt(product_name: str, category: str, description: str,
                                 specifications: str, layout_plan: dict) -> str:
    """Prompt for the deterministic pipeline's first step: a clean product
    shot (no arrows/labels at all) suitable for automatic corner detection.
    Deliberately separate from build_generation_prompt (rather than another
    branch inside it) so the existing, proven arrow-drawing prompt path is
    never at risk of being disturbed by this new one.
    """
    is_boxed = (is_boxed_multipiece_product(category, product_name, description, specifications)
               or is_typically_boxed_single_item(category, product_name, description, specifications))
    shape_note = ""
    if is_boxed:
        shape_note = (
            " This product is normally sold sealed in retail box/blister "
            "packaging — show it in its CLOSED, SEALED packaging (as it "
            "would look on a store shelf), as ONE single solid consolidated "
            "shape, not the loose item(s) removed, opened, or scattered "
            "around it — automatic corner detection needs one clean solid "
            "silhouette, not several separate pieces with gaps between them."
        )
    orientation = (layout_plan or {}).get("orientation", "three_quarter_corner")
    if orientation == "side_profile":
        camera_note = (
            " Camera: strict SIDE-PROFILE view, positioned directly to the "
            "side and perpendicular to the product's length, so its full "
            "silhouette is visible edge-on with no foreshortening."
        )
    elif orientation == "flat_front_on":
        camera_note = " Camera: flat, face-on view, product's full front face parallel to the camera."
    else:
        camera_note = (
            " Camera: 3/4 CORNER perspective — position the camera so ONE "
            "bottom corner of the product is closest to the viewer, with "
            "both adjacent bottom edges receding away from that corner at "
            "an angle, and the vertical edge rising from that same corner "
            "also visible (a classic product-dimension-diagram angle)."
        )
    context = f"Product: {product_name} (category: {category})"
    if description:
        context += f"\nDescription: {description[:600]}"
    if specifications:
        context += f"\nSpecifications: {specifications[:600]}"
    return (
        f"{context}\n\n"
        f"STRICT RULE: Preserve the exact product from the reference photo "
        f"— same shape, colors, materials, printed artwork, and proportions. "
        f"Do not invent, add, or remove any part.{shape_note}{camera_note}\n\n"
        f"{CLEAN_PRODUCT_BACKGROUND_RULE}\n\n"
        f"{CLEAN_PRODUCT_FILL_FRAME_RULE}\n\n"
        f"Composition: 1:1 square canvas, high resolution, realistic "
        f"photography, clean professional catalog style, no watermark."
    )


def _detect_product_corners_cv_worker(image_bytes: bytes, conn) -> None:
    """Runs in a separate PROCESS (see detect_product_corners_cv) — never
    called directly."""
    try:
        result = _detect_product_corners_cv_impl(image_bytes)
    except Exception:
        result = None
    try:
        conn.send(result)
    except Exception:
        pass
    finally:
        conn.close()


def detect_product_corners_cv(image_bytes: bytes, _timeout_seconds: float = 20.0) -> "dict | None":
    """Classical CV corner detection on a clean (no-arrows) product image.
    Returns {"polygon": [[x,y],...], "near_idx": i, ...} on a confident
    detection, or None if any sanity check fails — callers MUST treat None
    as "fall back to the existing model-drawn-arrows path", never guess.

    BUG FOUND (live test): an earlier version of this ran the work in a
    background THREAD with a timeout via concurrent.futures. That looked
    right but didn't actually work — cv2.grabCut apparently never releases
    the GIL during its C++ computation on some inputs (confirmed live: a
    car rendered small against a large flat background made grabCut run for
    minutes), so the MAIN thread waiting on future.result(timeout=20)
    couldn't even get scheduled to notice the timeout had elapsed until the
    background call finally finished on its own — the exact stall this was
    supposed to prevent. A separate PROCESS doesn't have this problem: the
    OS can forcibly terminate it after the timeout regardless of what its
    C code is doing internally, which is what an unreliable local
    computation step genuinely needs (same "don't trust it, verify it can
    actually be cut off" lesson as everything else in this pipeline).
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_detect_product_corners_cv_worker, args=(image_bytes, child_conn))
    proc.start()
    child_conn.close()
    proc.join(timeout=_timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return None
    result = parent_conn.recv() if parent_conn.poll() else None
    parent_conn.close()
    return result


def _detect_product_corners_cv_impl(image_bytes: bytes) -> "dict | None":
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    frame_area = w * h

    # GrabCut (graph-cut foreground/background segmentation) rather than a
    # brightness threshold or flood-fill — both of those broke on a real
    # generated image (a vignette background darker than assumed fooled a
    # fixed threshold; flood-fill from the corners leaked straight through
    # the product's OWN white packaging patches into the background on the
    # far side). GrabCut models foreground/background color distributions
    # instead of a purely local brightness rule, so it isn't fooled by an
    # isolated light patch inside the product.
    margin_frac = 0.03
    rect = (int(w * margin_frac), int(h * margin_frac),
           int(w * (1 - 2 * margin_frac)), int(h * (1 - 2 * margin_frac)))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((h, w), np.uint8)
    try:
        cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    product_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                            255, 0).astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(product_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # Too large (>85%) means segmentation likely failed to separate a real
    # background at all (traced the frame/vignette, not the product) — a
    # real failure mode seen live. Too small (<3%) means the product wasn't
    # found / mask is noise.
    if area > 0.85 * frame_area or area < 0.03 * frame_area:
        return None
    x, y, bw, bh = cv2.boundingRect(largest)
    edge_margin = 3
    touches = sum([x <= edge_margin, y <= edge_margin,
                  (x + bw) >= w - edge_margin, (y + bh) >= h - edge_margin])
    if touches >= 3:
        # Bounding box hugging 3+ image edges is not a real, centered
        # product shot — almost certainly background leakage.
        return None
    # BUG FOUND (product 27731, live test): the prompt asked for the product
    # to be prominent, but a real generation still came back small and
    # centered with large empty margins on every side — technically correct
    # (right product, flat background, no arrows) but far too few real
    # pixels for precise corner detection, producing cramped/overlapping
    # arrows and labels in the final image. A prompt instruction alone isn't
    # a guarantee (the same lesson as SQUARE_CANVAS_RULE needing
    # imageConfig.aspectRatio behind it) — deterministically reject a
    # too-small product here too, not just too-large/leaked segmentation,
    # so a small render safely falls back to the proven model-drawn path
    # instead of shipping cramped, hard-to-read measurements.
    if max(bw, bh) < 0.45 * min(w, h):
        return None

    peri = cv2.arcLength(largest, True)
    approx = None
    for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        candidate = cv2.approxPolyDP(largest, eps_frac * peri, True)
        if 4 <= len(candidate) <= 8:
            approx = candidate
            break
    if approx is None or len(approx) < 4:
        return None
    pts = approx.reshape(-1, 2)
    n = len(pts)
    near_idx = int(np.argmax(pts[:, 1]))
    return {
        "polygon": pts.tolist(), "near_idx": near_idx,
        "prev_idx": (near_idx - 1) % n, "next_idx": (near_idx + 1) % n,
        "image_w": w, "image_h": h,
    }


def _edge_angle_deg(p1, p2) -> float:
    """0 = perfectly horizontal, 90 = perfectly vertical (unsigned acute
    angle from horizontal, regardless of which quadrant the edge points
    into) — atan2's raw range is (-180, 180], so an edge pointing into the
    third quadrant (both dx and dy negative) came back as e.g. -150 degrees,
    whose abs() (150) is NOT the acute angle from horizontal (it's actually
    a fairly horizontal 30-degree edge pointing the other way) — folding
    anything over 90 back down is required, not optional."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return 180 - angle if angle > 90 else angle


def assign_axes_from_corners(corner_info: dict) -> "dict | None":
    """Maps the detected polygon around the near corner to length/breadth
    (ground-plane, roughly horizontal) and height (roughly vertical) pixel
    edges. Returns None (fall back) rather than guessing if the polygon
    doesn't resolve into a confident 2-horizontal + 1-vertical set — a 3D
    box corner has 3 real edges, but a 2D silhouette vertex only ever shows
    2 adjacent boundary edges, so the third (usually height, for a box
    lying with a small height) has to be found one hop further around the
    polygon, not assumed to touch the near corner directly.
    """
    pts = corner_info["polygon"]
    n = len(pts)
    near_idx = corner_info["near_idx"]
    near = pts[near_idx]

    candidates = []
    for start_off, end_off in ((0, 1), (0, -1), (1, 2), (-1, -2)):
        i1 = (near_idx + start_off) % n
        i2 = (near_idx + end_off) % n
        if i1 == i2:
            continue
        p1, p2 = pts[i1], pts[i2]
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if length < 15:
            continue
        angle = _edge_angle_deg(p1, p2)
        d1 = np.hypot(p1[0] - near[0], p1[1] - near[1])
        d2 = np.hypot(p2[0] - near[0], p2[1] - near[1])
        near_end, far_end = (p1, p2) if d1 < d2 else (p2, p1)
        candidates.append({"near_end": near_end, "far_end": far_end,
                           "length": length, "angle": angle})

    horiz = sorted((c for c in candidates if c["angle"] < 55), key=lambda c: -c["length"])
    vert = sorted((c for c in candidates if c["angle"] >= 55), key=lambda c: -c["length"])
    if len(horiz) < 2 or len(vert) < 1:
        return None
    ground_sorted = sorted(horiz[:2], key=lambda c: -c["length"])
    height_edge = vert[0]
    # BUG FOUND (product 35806, live test): a 2-hop candidate edge (needed
    # for height on a box whose near corner has no directly-attached
    # vertical silhouette edge) does NOT touch the true near corner at all
    # — its own "near_end" is just whichever of ITS OWN two endpoints is
    # closer to the near corner, not the near corner itself. Returning only
    # a single shared "near_corner" and forcing every axis's arrow to start
    # there drew a bogus diagonal line cutting straight across the box for
    # any 2-hop axis, completely missing its real edge. Each axis now
    # carries its OWN near/far pair from its own candidate edge — 3
    # independent arrows, each following one real product edge, rather than
    # 3 arrows forced to fan out from one single shared point.
    return {
        "length": {"near": list(map(int, ground_sorted[0]["near_end"])),
                  "far": list(map(int, ground_sorted[0]["far_end"])),
                  "px": ground_sorted[0]["length"]},
        "breadth": {"near": list(map(int, ground_sorted[1]["near_end"])),
                   "far": list(map(int, ground_sorted[1]["far_end"])),
                   "px": ground_sorted[1]["length"]},
        "height": {"near": list(map(int, height_edge["near_end"])),
                  "far": list(map(int, height_edge["far_end"])),
                  "px": height_edge["length"]},
    }


def draw_dimension_arrows_deterministic(image_path: str, axes: dict, dim_data: dict) -> None:
    """Overwrites image_path in place with deterministically-drawn
    measurement arrows + labels, using REAL detected pixel coordinates
    (axes) and REAL admin numbers (dim_data) — no generative model text
    rendering involved anywhere, so no swapped axes, no misspelled labels,
    no corrupted numbers (all defects the model-drawn approach hit)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Pillow's built-in default font (load_default(size=...), Pillow >=10.1),
    # not a named system font path — a hardcoded macOS font path here would
    # simply not exist on the GCP Linux instance this actually runs on (same
    # reasoning as add_size_disclaimer already documents).
    font_size = max(24, img.size[0] // 40)
    font = ImageFont.load_default(size=font_size)
    unit = dim_data.get("unit") or "cm"

    img_cx, img_cy = img.size[0] / 2, img.size[1] / 2

    def _draw_one(axis, label_name, value):
        if value is None or axis is None:
            return
        near = tuple(axis["near"])
        far = tuple(axis["far"])
        draw.line([near, far], fill=(20, 20, 20), width=4)
        vx, vy = far[0] - near[0], far[1] - near[1]
        norm = max((vx**2 + vy**2) ** 0.5, 1e-6)
        vx, vy = vx / norm, vy / norm
        perp = (-vy, vx)
        for end, direction in ((near, 1), (far, -1)):
            tip = end
            base1 = (end[0] - vx * direction * 14 + perp[0] * 7,
                    end[1] - vy * direction * 14 + perp[1] * 7)
            base2 = (end[0] - vx * direction * 14 - perp[0] * 7,
                    end[1] - vy * direction * 14 - perp[1] * 7)
            draw.polygon([tip, base1, base2], fill=(20, 20, 20))

        # Offset the label perpendicular to the arrow line (not centered ON
        # it) with a solid white background box behind the text. A real
        # render (product 2145, a wide/thin 24x4x18cm box) had labels
        # sitting almost on top of the product itself: the OLD fixed 28px
        # offset is negligible on a 2048px canvas, and always used the same
        # perpendicular direction regardless of which side of the line the
        # product actually was on. Now the offset scales with image size
        # (matching font_size's own scaling) and the perpendicular
        # direction is chosen to point AWAY from the image center — since
        # the product occupies the frame's middle, pushing the label
        # toward whichever side is farther from center reliably lands it
        # in open background space instead of over the product.
        mid = ((near[0] + far[0]) / 2, (near[1] + far[1]) / 2)
        if (mid[0] - img_cx) * perp[0] + (mid[1] - img_cy) * perp[1] < 0:
            perp = (-perp[0], -perp[1])
        offset = max(60, img.size[0] // 12)
        text_x = mid[0] + perp[0] * offset
        text_y = mid[1] + perp[1] * offset
        text = f"{label_name} {value:g} {unit}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_x = text_x - text_w / 2
        box_y = text_y - text_h / 2
        pad = 6
        # The larger offset above pushes labels further out, which can
        # otherwise carry a label past the canvas edge for a product framed
        # close to the frame border — clamp so the label (with its padding)
        # always stays fully on-canvas.
        box_x = max(pad, min(box_x, img.size[0] - text_w - pad))
        box_y = max(pad, min(box_y, img.size[1] - text_h - pad))
        draw.rectangle([box_x - pad, box_y - pad, box_x + text_w + pad, box_y + text_h + pad],
                       fill=(255, 255, 255))
        draw.text((box_x, box_y), text, fill=(20, 20, 20), font=font)

    _draw_one(axes.get("length"), "Length", dim_data.get("length"))
    _draw_one(axes.get("breadth"), "Breadth", dim_data.get("breadth"))
    _draw_one(axes.get("height"), "Height", dim_data.get("height"))
    img.save(image_path)


def try_deterministic_dimension_image(reference_url: str, product_name: str, category: str,
                                      description: str, specifications: str, layout_plan: dict,
                                      dim_data: dict, out_path: str, project_id: str, region: str,
                                      tokens: "VertexTokenProvider") -> bool:
    """Attempts the full deterministic pipeline for one dimension slot.
    Returns True and leaves a finished, disclaimer-free dimension image at
    out_path on confident success; False (out_path's contents undefined) on
    ANY uncertainty, in which case the caller falls back to the existing
    model-drawn-arrows + verify/retry system unchanged.
    """
    prompt = build_clean_dimension_prompt(product_name, category, description,
                                         specifications, layout_plan)
    result = generate_image(reference_url, prompt, out_path, project_id, region, tokens)
    if result["status"] != "generated":
        return False
    try:
        with open(out_path, "rb") as f:
            image_bytes = f.read()
    except OSError:
        return False
    shape_check = check_square_and_min_px(image_bytes, MIN_OUTPUT_PX)
    if not shape_check["valid"]:
        return False
    corners = detect_product_corners_cv(image_bytes)
    if corners is None:
        return False
    axes = assign_axes_from_corners(corners)
    if axes is None:
        return False
    try:
        draw_dimension_arrows_deterministic(out_path, axes, dim_data)
    except Exception:
        return False
    return True


MAX_GENERATION_ATTEMPTS = 3


def build_retry_feedback_text(failure_categories: list, reason: str, layout_plan: dict) -> str:
    """Turns a SPECIFIC QC failure into a targeted correction instruction for
    the next attempt (section 14) instead of resubmitting an identical
    prompt three times in a row."""
    lines = [f"PREVIOUS QC FAILURE: {reason}", "", "REQUIRED CORRECTION:"]
    cats = set(failure_categories or [])
    if FAILURE_AXIS_SWAP in cats or FAILURE_GEOMETRY_MISMATCH in cats:
        mapping = (layout_plan or {}).get("dimension_mapping", {})
        lines += [
            "- Preserve the product exactly as in the reference image.",
            "- Do not swap or change the numerical values themselves.",
            f"- Reorient the product so its {(layout_plan or {}).get('length_axis', 'longest physical axis')} "
            f"is the visually dominant, longest axis, carrying the Length value "
            f"({mapping.get('length', '')}).",
            f"- Keep {mapping.get('breadth', '')} on the shorter breadth/depth axis "
            f"({(layout_plan or {}).get('breadth_axis', '')}).",
            f"- Keep {mapping.get('height', '')} on the thickness axis "
            f"({(layout_plan or {}).get('height_axis', '')}).",
        ]
    if FAILURE_ARROW_COUNT in cats:
        lines.append(
            "- Focus on annotation placement: draw exactly the arrows specified "
            "above, each entirely in empty background space, none crossing or "
            "overlapping the product or each other."
        )
    if FAILURE_TEXT_MISMATCH in cats:
        lines.append(
            "- Re-render every measurement label's text exactly as given — double "
            "check spelling and the exact number/unit before finalizing."
        )
    if FAILURE_MISSING_OR_DUPLICATE in cats:
        lines.append(
            "- You drew two separate arrows for the SAME measurement name and "
            "completely omitted a different required one (see PREVIOUS QC "
            "FAILURE above for exactly which). Draw exactly one arrow per "
            "named measurement — Length, Breadth, and Height must each appear "
            "on its own distinct arrow, no name skipped, no name repeated."
        )
    if FAILURE_SCALE_MISMATCH in cats:
        lines.append(
            "- The product was rendered at the wrong real-world scale relative to "
            "the person/hand/object in this scene — re-render it noticeably "
            "smaller or larger (per the failure reason above) so its size "
            "relative to that person/object genuinely matches its real listed "
            "dimensions, not an enlarged or shrunk 'hero' size."
        )
    if FAILURE_FEATURE_LABEL_SPELLING in cats:
        lines.append(
            "- The text label rendered on this image was misspelled or garbled (see "
            "PREVIOUS QC FAILURE above for the exact wrong text). Re-render the label "
            "text so it is spelled correctly, letter by letter, with no dropped, "
            "doubled, or swapped characters — double check it before finalizing."
        )
    if FAILURE_FEATURE_POINTER_WRONG in cats:
        lines.append(
            "- The pointer/leader line from the text label did not land on the correct "
            "part (see PREVIOUS QC FAILURE above for which feature). Re-render so the "
            "pointer's tip touches the SAME physical part named by the label, with "
            "nothing else nearby it could be mistaken for — reframe as a tighter "
            "close-up crop on that exact part if needed so there is no ambiguity about "
            "which part the pointer is indicating."
        )
    if FAILURE_PACKAGING_IN_ANGLE_SHOT in cats:
        lines.append(
            "- This shot showed the retail box/packaging instead of (or alongside) the "
            "bare product. Re-render showing ONLY the product itself, fully removed "
            "from any box or blister pack, rotated to the requested view — no "
            "packaging anywhere in the frame."
        )
    if FAILURE_MISSING_PACKED_OR_UNPACKED in cats:
        lines.append(
            "- This shot must show BOTH the real retail package AND the unpacked "
            "product together, and one of the two was missing, hidden, or the box "
            "looked empty/generic (see PREVIOUS QC FAILURE above). Re-render with a "
            "genuine, intact, non-empty box (its real printed artwork) placed next "
            "to the complete product fully removed from that box — both clearly "
            "visible, neither hiding the other."
        )
    if FAILURE_PRODUCT_AUTHENTICITY in cats:
        lines.append(
            "- The product itself was wrong: either it no longer matches the "
            "reference photo's real shape/color/parts/branding, or it now shows "
            "a part, accessory, or design element the reference photo and "
            "product facts do not support. Re-render the EXACT same physical "
            "product as the reference image — same shape, proportions, color, "
            "material, and parts — and remove anything invented."
        )
    if len(lines) == 3:
        lines.append("- Address the specific problem described in PREVIOUS QC FAILURE above.")
    return "\n".join(lines)


DIMENSION_AUDIT_LOG_PATH = os.environ.get("DIMENSION_AUDIT_LOG", "dimension_audit_log.jsonl")
_dimension_audit_lock = threading.Lock()


def log_dimension_audit(entry: dict) -> None:
    """One JSON line per dimension-slot outcome (section 17) — product_id,
    dimension_status, layout plan, geometry analysis, QC verdicts per
    attempt, final status. Thread-safe (concurrent slot tasks) and
    best-effort — a logging failure must never break the actual run."""
    try:
        with _dimension_audit_lock:
            with open(DIMENSION_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


def generate_image_with_verification(reference_url: str, prompt: str, out_path: str,
                                     project_id: str, region: str, tokens: VertexTokenProvider,
                                     axis_labels: list, is_vehicle: bool = False,
                                     text_only_label: str = "", product_name: str = "",
                                     description: str = "", specifications: str = "",
                                     image_type: str = "", forced_variation: str = "",
                                     exclude_features: list = None,
                                     retry_prompt_fn=None, attempt_log_fn=None,
                                     known_length_cm: float = None) -> dict:
    """Wraps generate_image with the verify-and-retry loop, regenerating up
    to MAX_GENERATION_ATTEMPTS (3, capped — real cost per attempt) total
    times until verification passes. Exhausting all 3 without a pass keeps
    the LAST attempt's file (better than nothing) but reports
    "generated_unverified" so it can be flagged for manual review instead
    of silently shipped as a clean pass.

    Every attempt is checked two ways regardless of slot type:
      1. check_square_and_min_px — deterministic 1:1 / >=2048px gate.
      2. Content verification — verify_dimension_image for a slot with
         checkable ground truth (axis_labels non-empty), otherwise
         verify_generated_image_generic (product fidelity / invented
         parts / unsupported or duplicate feature / real-world scale vs
         known_length_cm for Lifestyle slots) for every other slot.
         Previously non-dimension slots skipped this step entirely.

    retry_prompt_fn(failure_categories, reason, attempt) -> str, if given,
    is called before every attempt AFTER the first to build a prompt that
    responds to the SPECIFIC previous failure (e.g. a targeted correction
    for LENGTH_BREADTH_AXIS_SWAP, a different one for PRODUCT_DISTORTION)
    instead of resubmitting the identical prompt three times. attempt_log_fn
    (product_id, attempt, verdict_dict), if given, is called after every
    attempt for audit logging (see log_dimension_audit) — best-effort, never
    allowed to raise past this function.
    """
    last_result = None
    last_failure_categories = []
    last_reason = ""
    reference_bytes = None
    reference_content_type = None
    current_prompt = prompt
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        if attempt > 1 and retry_prompt_fn:
            current_prompt = retry_prompt_fn(last_failure_categories, last_reason, attempt)
        result = generate_image(reference_url, current_prompt, out_path, project_id, region, tokens)
        if result["status"] != "generated":
            last_result = result
            continue
        with open(out_path, "rb") as f:
            image_bytes = f.read()
        shape_check = check_square_and_min_px(image_bytes, MIN_OUTPUT_PX)
        if not shape_check["valid"]:
            last_failure_categories, last_reason = ["OTHER"], shape_check["reason"]
            last_result = {"status": f"generated_unverified: {shape_check['reason']}",
                           "failure_categories": last_failure_categories}
            if attempt_log_fn:
                attempt_log_fn(attempt, {"valid": False, "reason": shape_check["reason"],
                                        "failure_categories": ["OTHER"]})
            continue
        if axis_labels:
            verdict = verify_dimension_image(image_bytes, axis_labels, project_id, region, tokens,
                                            is_vehicle=is_vehicle, text_only_label=text_only_label)
        else:
            if reference_bytes is None:
                try:
                    reference_bytes, reference_content_type = get_image_bytes(reference_url)
                except (requests.RequestException, OSError):
                    reference_bytes, reference_content_type = b"", ""
            if reference_bytes:
                verdict = verify_generated_image_generic(
                    image_bytes, reference_bytes, reference_content_type, product_name,
                    description, specifications, image_type, forced_variation,
                    exclude_features, project_id, region, tokens,
                    known_length_cm=known_length_cm)
            else:
                # Couldn't fetch the reference to compare against — fail
                # open rather than block the row on a network hiccup.
                verdict = {"valid": True, "reason": ""}
        if attempt_log_fn:
            attempt_log_fn(attempt, verdict)
        if verdict["valid"]:
            return result
        last_failure_categories = verdict.get("failure_categories", [])
        last_reason = verdict.get("reason", "")
        last_result = {"status": f"generated_unverified: {verdict['reason']}",
                       "failure_categories": last_failure_categories}
    # Ran out of attempts — last_result is either a real generation failure
    # or "generated_unverified" (file on disk, just never passed the check),
    # in which case it carries failure_categories from the LAST attempt so a
    # caller (process_slot_task's Lifestyle scale/authenticity fallback) can
    # act on WHICH check kept failing, not just that something did.
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
        image_urls = split_image_urls(row.get("Image_URLs", ""))
        reference_url = pick_reference_image(covered, image_urls)
        try:
            reference_quality = json.loads(row.get("Reference_Quality") or "{}")
        except (json.JSONDecodeError, TypeError):
            # Absent/older classification_result.csv without this column —
            # build_reference_set treats an empty dict as "no quality data,
            # trust every covered image", i.e. exactly the pre-existing
            # behavior, never a crash.
            reference_quality = {}
        reference_set = build_reference_set(covered, slots, image_urls, reference_quality)
        slot_variations, feature_positions = assign_slot_variations(slots, missing_slot_nums)

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
                "reference_set": reference_set,
                "rule_category": rule_category,
                "product_id": str(row["Product_ID"]),
                "sku": row["SKU"],
                "product_name": row["Name"],
                "description": row.get("Description", ""),
                "specifications": row.get("Specifications", ""),
                "forced_variation": slot_variations.get(slot_num, ""),
                "feature_position": feature_positions.get(slot_num),
                "image_out_dir": image_out_dir,
                # True only when NO existing image was classified as
                # covering ANY slot for this product — the only case where
                # pick_reference_image() falls back to the first raw image
                # URL with zero validation (previously silent; see
                # ReferenceAssessmentCache/assess_reference_image in
                # process_slot_task, which specifically gates THIS path).
                "reference_unverified": not covered,
                # Raw, unfiltered list of every admin photo for this product
                # (not just ones classify_images.py matched to a slot) — the
                # packed+unpacked combo slot needs to search ALL of them for
                # a genuine boxed shot, since a real boxed photo can exist on
                # admin even when classify_images.py rejected it for every
                # slot (that's exactly what happened for product 2161).
                "image_urls": image_urls,
            })
    return tasks


def process_slot_task(task: dict, project_id: str, region: str, tokens: VertexTokenProvider,
                      uploader: "GcsUploader | None", upload_cache: "UploadCache",
                      overwrite: bool, feature_cache: "FeatureExtractionCache" = None,
                      reference_cache: "ReferenceAssessmentCache" = None) -> dict:
    """Does the real work for ONE (product, slot) task and returns a
    final_output row, tagged with an internal "_bucket"/"_needs_review" for
    the run-summary counts (stripped from what the reader sees — write_final_excel
    only looks up specific named columns, so these extra keys are harmless).

    Called from worker threads (see main()'s use of run_concurrent) —
    everything it touches (uploader, upload_cache, tokens) is already
    safe under concurrency: UploadCache/maybe_upload only ever mutate via
    a single dict-set + json.dump per call (fine — worst case is a
    redundant re-upload, not corruption), VertexTokenProvider now locks
    its own refresh, and FeatureExtractionCache is likewise safe to share
    across a product's sibling feature-slot tasks.
    """
    row_base = task["row_base"]
    if task["kind"] == "no_rule":
        return {**row_base, "Status": "no_rule_for_category", "_bucket": "Failed"}

    slot_num = task["slot_num"]
    slot_info = task["slot_info"]
    image_type_lower = slot_info["image_type"].lower()
    is_packed_and_unpacked_combo_slot = ("unpacked and packed" in image_type_lower
                                        or "packed and unpacked" in image_type_lower)

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

    # Per-slot reference — a "Box Back Content" slot uses a real packaging
    # photo when this product has one covering some packaging-type slot, a
    # "Feature" slot uses a real feature-type photo, etc., instead of every
    # slot reusing whatever single photo covers slot 1 regardless of what
    # it actually needs to depict (see build_reference_set). Falls back to
    # "primary" (== task["reference_url"]) when no better-suited photo
    # exists for this specific slot — never worse than the old behavior,
    # only better when the admin panel actually has a relevant photo.
    slot_reference_url = pick_slot_reference(task["reference_set"], slot_info) or task["reference_url"]

    low_quality_reference = False
    if reference_cache:
        # BUG FOUND (products 33955/39080 — admin's uploaded photo was a
        # completely different product under the same category): this check
        # used to only run when task["reference_unverified"] was set, i.e.
        # only when classify_images.py had ALREADY rejected every existing
        # image for this product. That made this safety net entirely
        # dependent on a DIFFERENT, less rigorous classifier's opinion —
        # classify_images.py only judges "does this photo fit slot N's
        # generic description", and on a run where it happened to accept
        # the wrong photo for even one slot, this identity check was never
        # even consulted, and a fundamentally wrong reference photo was used
        # to generate every other slot completely undetected (that's
        # exactly what shipped in the original 952-product batch). Now runs
        # unconditionally, once per product (ReferenceAssessmentCache still
        # dedupes it across that product's sibling slot tasks), so a wrong
        # reference photo is always caught, not just when another check's
        # per-run judgment happens to also catch it.
        assessment = reference_cache.get_or_assess(
            task["product_id"], task["reference_url"], task["product_name"],
            task["description"], task["specifications"], project_id, region, tokens)
        if assessment.get("defect_type") == "wrong_product_or_unusable":
            log_dimension_audit({
                "product_id": task["product_id"], "sku": task["sku"], "slot": slot_num,
                "image_type": slot_info["image_type"],
                "final_status": "MANUAL_REVIEW_REQUIRED",
                "reason": (f"Reference photo used for generation does not actually show "
                          f"this product: {assessment.get('reason', '')}. A corrected "
                          f"reference photo needs to be sourced/uploaded (e.g. verified "
                          f"against the product's real listing) before this product can "
                          f"be generated."),
            })
            # Image_Source carries the FLAGGED reference photo itself (not a
            # generated deliverable) so a reviewer has something to actually
            # click and look at — a reviewer can't act on a bare status word
            # with nothing behind it. build_wide_summary.py labels this
            # distinctly ("reference photo, not final") so it's never
            # mistaken for a passing image.
            return {**row_base, "Status": "MANUAL_REVIEW_REQUIRED",
                   "Image_Source": task["reference_url"],
                   "GCP_Link": "", "_bucket": "Failed", "_needs_review": True,
                   "_upload_failed": False}
        low_quality_reference = assessment.get("defect_type") == "quality_only"

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

    # "Full Unpacked and Packed Front-Angle View" — three different prompt
    # strategies all failed live (product 2161, a real doll whose admin
    # panel DOES have a genuine boxed photo): a single image-EDIT call
    # kept dropping the box entirely rather than rendering both a boxed AND
    # an unpacked instance together. Deterministic composite instead: if
    # any of this product's raw admin photos genuinely shows it boxed,
    # generate a plain unpacked photo separately and paste the two side by
    # side — both instances are then guaranteed present, not dependent on
    # one edit call's compositional ability. Falls through to the existing
    # single-call generative attempt (best effort) if no boxed photo can be
    # found, or if anything in this path fails.
    if is_packed_and_unpacked_combo_slot:
        boxed_url = next(
            (u for u in task.get("image_urls", [])
             if image_shows_packaged_product(u, project_id, region, tokens)),
            None,
        )
        if boxed_url:
            try:
                # FRONT-FACING, not "a different angle" — this panel sits
                # right next to the boxed panel (which itself faces front,
                # since that's how retail packaging is photographed) and
                # must visually match that orientation. A real run asked for
                # "a different view" here and got the doll shown from the
                # BACK, which reads as inconsistent/wrong next to a
                # front-facing box.
                unpacked_slot_info = {
                    **slot_info, "image_type": "Additional Angle",
                    "description": ("Show the complete UNBOXED product facing the "
                                    "camera directly, front-on, on a plain white "
                                    "background — the same front-facing orientation "
                                    "as the product would be seen through its own "
                                    "retail packaging, not from the back or a side "
                                    "angle."),
                }
                unpacked_prompt = build_generation_prompt(
                    task["product_name"], task["rule_category"], unpacked_slot_info,
                    task["description"], task["specifications"],
                    forced_variation=("the complete UNBOXED product facing the camera "
                                      "directly, front-on, on a plain white "
                                      "background — not from the back or a side angle"))
                unpacked_out_path = out_path + ".unpacked_tmp.png"
                unpacked_result = generate_image_with_verification(
                    boxed_url, unpacked_prompt, unpacked_out_path, project_id, region, tokens,
                    axis_labels=[], product_name=task["product_name"],
                    description=task["description"], specifications=task["specifications"],
                    image_type="Additional Angle")
                unpacked_status = unpacked_result["status"]
                if unpacked_status == "generated" or unpacked_status.startswith("generated_unverified"):
                    boxed_bytes, _ = get_image_bytes(boxed_url)
                    with open(unpacked_out_path, "rb") as f:
                        unpacked_bytes = f.read()
                    compose_packed_and_unpacked_image(boxed_bytes, unpacked_bytes, out_path)
                    os.remove(unpacked_out_path)
                    gcp_link, upload_failures = maybe_upload(uploader, upload_cache, filename, out_path, 0)
                    return {**row_base, "Status": "Generated", "Image_Source": filename,
                           "GCP_Link": gcp_link, "_bucket": "Generated",
                           "_upload_failed": upload_failures > 0}
            except Exception:
                pass

    forced_variation = task["forced_variation"]
    exclude_features = None
    no_feature_available = False
    if image_type_lower.startswith("feature") and task.get("feature_position") and feature_cache:
        idx, total = task["feature_position"]
        extracted = feature_cache.get_or_extract(
            task["product_id"], task["product_name"], task["rule_category"],
            task["description"], task["specifications"], total,
            project_id, region, tokens)
        if len(extracted) > idx:
            # Enough genuinely distinct features to cover every sibling
            # Feature slot for this product — assign this one its real,
            # concrete feature and exclude the others.
            forced_variation = extracted[idx]
            exclude_features = [f for i, f in enumerate(extracted) if i != idx]
        elif extracted:
            # Extraction worked but this product's own description/spec
            # text genuinely doesn't support as many distinct features as
            # this category has Feature slots — falling back to the old
            # generic rotation category here (e.g. "a specific included
            # accessory") is exactly how a fake second feature gets
            # invented, since the model has to pick SOMETHING. Use a plain
            # alternate product view for this slot instead of a feature
            # callout.
            no_feature_available = True
            forced_variation = (
                "a different full or three-quarter view of the complete "
                "product, clearly distinct from this product's other images"
            )
            exclude_features = extracted
        # else: extraction produced nothing usable at all — keep the
        # pre-existing generic rotation fallback (task["forced_variation"]),
        # unchanged from before, since we have no product-specific signal
        # either way.

    # A Box/Package-type slot's whole point is showing the product's REAL
    # retail packaging design — printed artwork, logo placement, colors,
    # text — none of which can be legitimately guessed. packaging_matches
    # (verify_generated_image_generic) only constrains this when the
    # REFERENCE photo itself already shows packaging; if this product has
    # no real packaging photo anywhere in its admin images (checked via
    # reference_set — see build_reference_set/pick_slot_reference), that
    # check is vacuously true and the model is free to invent an entire
    # box design from category-generic wording alone. A real defect
    # (product 35806) did exactly that — an invented Back-Panel design
    # with no basis in reality. Same policy as the dimension slot: don't
    # invent packaging with nothing to preserve fidelity from — fall back
    # to a plain, real product photo instead.
    is_box_slot = any(k in image_type_lower for k in _PACKAGING_ROLE_KEYWORDS)
    if is_box_slot and not task["reference_set"].get("packaging"):
        # Overriding ONLY "image_type" here would leave "description" as
        # this slot's ORIGINAL rule-master text (e.g. "Show the correct
        # retail package together with...") — build_generation_prompt
        # inserts slot_info['description'] verbatim as "Requirement: ..."
        # regardless of image_type, so a real run kept generating box
        # artwork anyway despite the "Additional Angle" relabeling. Both
        # keys must be overridden together.
        alt_slot_info = {**slot_info, "image_type": "Additional Angle",
                         "description": ("Show a clean, different full or three-quarter "
                                         "view of the complete UNBOXED product, distinct "
                                         "from this product's other images.")}
        alt_prompt = build_generation_prompt(
            task["product_name"], task["rule_category"], alt_slot_info,
            task["description"], task["specifications"],
            forced_variation=("a clean, different full or three-quarter view of the "
                              "complete UNBOXED product — a plain product photo, NOT "
                              "packaging/box artwork"),
            low_quality_reference=low_quality_reference)
        alt_result = generate_image_with_verification(
            slot_reference_url, alt_prompt, out_path, project_id, region, tokens,
            axis_labels=[], product_name=task["product_name"],
            description=task["description"], specifications=task["specifications"],
            image_type="Additional Angle")
        alt_status = alt_result["status"]
        alt_generated = alt_status == "generated" or alt_status.startswith("generated_unverified")
        gcp_link, upload_failed = "", False
        if alt_generated:
            gcp_link, upload_failures = maybe_upload(uploader, upload_cache, filename, out_path, 0)
            upload_failed = upload_failures > 0
        display_status = ("Generated" if alt_status == "generated"
                          else "Generated (needs review)" if alt_generated else alt_status)
        log_dimension_audit({
            "product_id": task["product_id"], "sku": task["sku"], "slot": slot_num,
            "image_type": slot_info["image_type"], "final_status": "NO_REAL_PACKAGING_REFERENCE",
            "reason": ("No admin photo anywhere for this product shows the real retail "
                      "packaging — generating a box/package design from scratch is not "
                      "attempted; a plain product photo was generated in its place."),
        })
        return {**row_base, "Status": display_status,
               "Image_Source": filename if alt_generated else "", "GCP_Link": gcp_link,
               "_bucket": "Generated" if alt_generated else "Failed",
               "_needs_review": alt_status.startswith("generated_unverified"),
               "_upload_failed": upload_failed}

    # Dimension-type slots have a checkable ground truth (an exact, named
    # set of measurements) IF AND ONLY IF the admin data actually proves
    # which number is which axis — classify_dimensions() is the gate that
    # decides that BEFORE any prompt is built or any API call is made (see
    # DIMENSION_STATUS_* in pipeline_lib.py). Never let the image model
    # guess an unproven axis order, and never invent numbers that don't
    # exist — both AMBIGUOUS and MISSING route straight to manual review.
    is_dimension_slot = "size" in image_type_lower or "dimension" in image_type_lower
    dim_data = None
    layout_plan = None
    geometry_analysis = None
    axis_labels = []
    is_vehicle = False
    text_only_label = ""

    if is_dimension_slot:
        # Rule (explicit product decision): if the admin's OWN reference
        # photo for this slot already IS a proper labeled dimension diagram
        # (printed numbers on arrows), that real photo is strictly more
        # trustworthy than anything we'd generate — use it as-is instead.
        # A dedicated check just for this slot, not the generic per-slot
        # classifier in classify_images.py, since a miss here ships a worse
        # result (an invented chart) than a miss on any other slot type.
        ref_check = reference_already_shows_dimensions(
            slot_reference_url, project_id, region, tokens)
        if ref_check.get("has_labeled_dimensions"):
            existing_filename = (f"{task['product_id']}_{slugify(task['sku'])}_{slot_num}_"
                                 f"{slugify(slot_info['image_type'])}_existing_dimensions.png")
            check = ensure_existing_image_quality(
                slot_reference_url, existing_filename, task["image_out_dir"],
                uploader, upload_cache, 0)
            log_dimension_audit({
                "product_id": task["product_id"], "sku": task["sku"], "slot": slot_num,
                "image_type": slot_info["image_type"], "final_status": check["status"],
                "reason": (f"Reused admin's own reference photo — already a labeled "
                          f"dimension diagram: {ref_check.get('reason', '')}"),
            })
            return {**row_base, "Status": check["status"], "Image_Source": check["image_source"],
                   "GCP_Link": check["gcp_link"], "_bucket": "Existing",
                   "_upload_failed": check["upload_failures"] > 0}

        # Explicit product decision (revised): this slot is no longer a
        # REQUIRED primary deliverable when the admin has no usable
        # dimension photo — repeated real-world hallucinations (wrong axis
        # order, invented numbers, distorted geometry) made a generated
        # chart risky enough that it must never occupy the slot a reviewer
        # expects to be a verified image. Instead:
        #   - The PRIMARY slot (this required slot number) is always a
        #     plain, easily-verifiable alternate-angle photo of the real
        #     product — never a dimension chart, never blank.
        #   - IF (and only if) the admin's own text data proves real,
        #     ordered measurements (VERIFIED status — a diecast scale
        #     estimate counts), a best-effort dimension chart is ALSO
        #     generated as a SEPARATE, clearly-labeled SECONDARY/bonus
        #     image alongside it — 7 images total for that product's slot
        #     count. If the data is AMBIGUOUS or MISSING, there is nothing
        #     legitimate to draw from at all, so no secondary attempt is
        #     made either — 6 images, same as any other slot.
        #   - The secondary chart is shown "as is", verified or not — it
        #     is explicitly a bonus, not a promised-accurate deliverable,
        #     so a reviewer decides whether it's usable rather than the
        #     pipeline silently shipping (or silently discarding) it.
        dim_data = classify_dimensions(task["description"], task["specifications"])
        if dim_data["status"] != DIMENSION_STATUS_VERIFIED:
            scale_ratio = detect_diecast_scale_ratio(
                task["rule_category"], task["product_name"],
                task["description"], task["specifications"])
            if scale_ratio:
                est = estimate_diecast_dimensions_cm(scale_ratio)
                dim_data = {
                    "length": est["length"], "breadth": est["breadth"],
                    "height": est["height"], "unit": "cm",
                    "source": f"Scale-derived (1:{scale_ratio} standard estimate, not admin data)",
                    "status": DIMENSION_STATUS_VERIFIED,
                    "raw_text": (f"Length {est['length']} cm, Breadth {est['breadth']} cm, "
                                f"Height {est['height']} cm"),
                    "confidence": "estimated",
                }

        secondary_row = None
        if dim_data["status"] == DIMENSION_STATUS_VERIFIED:
            layout_plan = build_measurement_layout_plan(
                dim_data, task["rule_category"], task["product_name"],
                task["description"], task["specifications"])
            try:
                ref_bytes, ref_ct = get_image_bytes(slot_reference_url)
                geometry_analysis = analyze_product_geometry(
                    ref_bytes, ref_ct, layout_plan, project_id, region, tokens)
            except (requests.RequestException, OSError):
                geometry_analysis = {}
            axis_labels = compute_axis_labels(task["description"], task["specifications"])
            if not axis_labels and dim_data.get("confidence") == "estimated":
                axis_labels = AXIS_LABEL_RE.findall(dim_data["raw_text"])
            is_vehicle = layout_plan["product_type"] == "vehicle"
            axis_labels, text_only_label = resolve_dimension_layout(axis_labels, is_vehicle)

            dim_prompt = build_generation_prompt(
                task["product_name"], task["rule_category"], slot_info,
                task["description"], task["specifications"], forced_variation,
                exclude_features, no_feature_available=no_feature_available,
                layout_plan=layout_plan, geometry_analysis=geometry_analysis,
                low_quality_reference=low_quality_reference)
            dim_attempt_log = []

            def _log_dim_attempt(attempt_num, verdict):
                dim_attempt_log.append({"attempt": attempt_num, "valid": verdict.get("valid"),
                                        "reason": verdict.get("reason", ""),
                                        "failure_categories": verdict.get("failure_categories", [])})

            def dim_retry_prompt_fn(failure_categories, reason, attempt):
                feedback = build_retry_feedback_text(failure_categories, reason, layout_plan)
                return build_generation_prompt(
                    task["product_name"], task["rule_category"], slot_info,
                    task["description"], task["specifications"], forced_variation,
                    exclude_features, no_feature_available=no_feature_available,
                    layout_plan=layout_plan, geometry_analysis=geometry_analysis,
                    retry_feedback=feedback)

            dim_filename = (f"{task['product_id']}_{slugify(task['sku'])}_{slot_num}_"
                            f"{slugify(slot_info['image_type'])}_secondary.png")
            dim_out_path = os.path.join(task["image_out_dir"], dim_filename)
            used_deterministic = False
            try:
                used_deterministic = try_deterministic_dimension_image(
                    slot_reference_url, task["product_name"], task["rule_category"],
                    task["description"], task["specifications"], layout_plan, dim_data,
                    dim_out_path, project_id, region, tokens)
            except Exception:
                used_deterministic = False
            if used_deterministic:
                dim_attempt_log.append({"attempt": 1, "valid": True,
                                        "reason": "deterministic_opencv_pipeline_success",
                                        "failure_categories": []})
                dim_result = {"status": "generated"}
            else:
                dim_result = generate_image_with_verification(
                    slot_reference_url, dim_prompt, dim_out_path, project_id, region, tokens,
                    axis_labels, is_vehicle=is_vehicle, text_only_label=text_only_label,
                    product_name=task["product_name"], description=task["description"],
                    specifications=task["specifications"], image_type=slot_info["image_type"],
                    forced_variation=forced_variation, exclude_features=exclude_features,
                    retry_prompt_fn=dim_retry_prompt_fn, attempt_log_fn=_log_dim_attempt)
            dim_status = dim_result["status"]
            dim_generated = dim_status == "generated" or dim_status.startswith("generated_unverified")
            if dim_generated:
                try:
                    add_size_disclaimer(dim_out_path)
                except Exception:
                    pass
                dim_link, dim_upload_failures = maybe_upload(
                    uploader, upload_cache, dim_filename, dim_out_path, 0)
                secondary_row = {
                    **row_base,
                    "Slot": f"{slot_num}b",
                    "Image_Type": f"{slot_info['image_type']} (secondary — verify before use)",
                    "Status": "Generated" if dim_status == "generated" else "Generated (needs review)",
                    "Image_Source": dim_filename, "GCP_Link": dim_link,
                    "_bucket": "Generated", "_needs_review": dim_status != "generated",
                    "_upload_failed": dim_upload_failures > 0,
                }
            log_dimension_audit({
                "product_id": task["product_id"], "sku": task["sku"], "slot": slot_num,
                "image_type": slot_info["image_type"], "dimension_status": dim_data["status"],
                "dimensions": {"length": dim_data.get("length"), "breadth": dim_data.get("breadth"),
                              "height": dim_data.get("height"), "unit": dim_data.get("unit")},
                "dimension_source": dim_data.get("source", ""),
                "attempts": dim_attempt_log,
                "final_status": ("SECONDARY_" + dim_status.upper().split(":")[0]) if dim_generated
                                else "SECONDARY_GENERATION_FAILED",
            })
        else:
            log_dimension_audit({
                "product_id": task["product_id"], "sku": task["sku"], "slot": slot_num,
                "image_type": slot_info["image_type"], "dimension_status": dim_data["status"],
                "raw_dimension_text": dim_data.get("raw_text", ""),
                "final_status": "NO_SECONDARY_ATTEMPT_NO_USABLE_DIMENSION_DATA",
                "reason": ("No axis-order guarantee in the admin data (numbers exist but "
                          "order to Length/Breadth/Height is unproven)"
                          if dim_data["status"] == DIMENSION_STATUS_AMBIGUOUS
                          else "No usable dimension data at all"),
            })

        # PRIMARY slot: a plain, easily-verifiable alternate-angle photo —
        # this required slot number never carries a dimension chart when
        # the admin has no usable photo for it, secondary attempt or not.
        # CONFIRMED real defect (product 2145): overriding only "image_type"
        # left "description" as this slot's ORIGINAL rule-master text
        # ("Show the product's relevant length, width, height... using
        # clear measurement lines") — build_generation_prompt inserts
        # slot_info['description'] verbatim as "Requirement: ..." regardless
        # of image_type, so the model kept drawing a dimension chart anyway.
        # Both keys must be overridden together.
        alt_slot_info = {**slot_info, "image_type": "Additional Angle",
                         "description": ("Show a clean, different full or three-quarter "
                                         "view of the complete product, distinct from "
                                         "this product's other images. Do NOT include "
                                         "any measurement lines, arrows, numbers, or "
                                         "dimension/size text of any kind.")}
        alt_prompt = build_generation_prompt(
            task["product_name"], task["rule_category"], alt_slot_info,
            task["description"], task["specifications"],
            forced_variation=("a clean, different full or three-quarter product "
                              "view — a plain product photo, NOT a dimension/"
                              "measurement diagram"),
            low_quality_reference=low_quality_reference)
        alt_result = generate_image_with_verification(
            slot_reference_url, alt_prompt, out_path, project_id, region, tokens,
            axis_labels=[], product_name=task["product_name"],
            description=task["description"], specifications=task["specifications"],
            image_type="Additional Angle")
        alt_status = alt_result["status"]
        alt_generated = alt_status == "generated" or alt_status.startswith("generated_unverified")
        gcp_link, upload_failed = "", False
        if alt_generated:
            gcp_link, upload_failures = maybe_upload(uploader, upload_cache, filename, out_path, 0)
            upload_failed = upload_failures > 0
        if alt_status == "generated":
            display_status = "Generated"
        elif alt_status.startswith("generated_unverified"):
            display_status = "Generated (needs review)"
        else:
            display_status = alt_status
        return {**row_base, "Status": display_status,
               "Image_Source": filename if alt_generated else "", "GCP_Link": gcp_link,
               "_bucket": "Generated" if alt_generated else "Failed",
               "_needs_review": alt_status.startswith("generated_unverified"),
               "_upload_failed": upload_failed, "_bonus_row": secondary_row}

    # Lifestyle images have no checkable "ground truth" the way a
    # Size/Dimensions slot does, but they DO have a real, checkable scale
    # problem: a real Hot Wheels car (~7.5 cm) came back looking ~18-20 cm
    # in a lifestyle scene — the "Scale check" note in build_generation_prompt
    # asked the model to keep it realistic, but nothing ever verified it
    # actually did. MISSING data (no numbers at all) still gets no check —
    # there is nothing to check against — but AMBIGUOUS data (real numbers,
    # unproven axis order) is fine to use here via
    # _ambiguous_longest_dimension_cm: this only needs the longest edge's
    # magnitude, never which axis it is, so the same "don't guess the axis
    # order" restriction that (correctly) blocks the labeled arrow chart
    # does not apply. Unlike is_dimension_slot, unverified data here just
    # means no extra check runs, not a hard MANUAL_REVIEW_REQUIRED gate,
    # since the measurement isn't this image's whole point.
    known_length_cm = None
    is_lifestyle_slot = "lifestyle" in image_type_lower
    if is_lifestyle_slot and not is_soft_foldable_product(
            task["rule_category"], task["product_name"], task["description"], task["specifications"]):
        lifestyle_dim_data = classify_dimensions(task["description"], task["specifications"])
        if lifestyle_dim_data["status"] == DIMENSION_STATUS_VERIFIED:
            longest = max(
                (v for v in (lifestyle_dim_data.get("length"), lifestyle_dim_data.get("breadth"),
                            lifestyle_dim_data.get("height")) if v is not None),
                default=None)
            known_length_cm = _to_cm(longest, lifestyle_dim_data.get("unit", ""))
        elif lifestyle_dim_data["status"] == DIMENSION_STATUS_AMBIGUOUS:
            known_length_cm = _ambiguous_longest_dimension_cm(lifestyle_dim_data.get("raw_text", ""))

    prompt = build_generation_prompt(task["product_name"], task["rule_category"], slot_info,
                                     task["description"], task["specifications"],
                                     forced_variation, exclude_features,
                                     no_feature_available=no_feature_available,
                                     layout_plan=layout_plan, geometry_analysis=geometry_analysis,
                                     low_quality_reference=low_quality_reference)

    attempt_log = []

    def _log_attempt(attempt_num, verdict):
        attempt_log.append({"attempt": attempt_num, "valid": verdict.get("valid"),
                            "reason": verdict.get("reason", ""),
                            "failure_categories": verdict.get("failure_categories", [])})

    # Every generic-verifier check (duplicate instance, authenticity,
    # packaging-in-angle-shot, scale, feature label/pointer, ...) can now
    # fire on any slot type, not just dimension/lifestyle/feature — a
    # targeted retry needs this for every slot, not a subset of slot
    # types (build_retry_feedback_text already degrades gracefully to a
    # generic "address the problem" line for a category it doesn't
    # recognize, so this is never worse than before for any slot).
    def retry_prompt_fn(failure_categories, reason, attempt):
        feedback = build_retry_feedback_text(failure_categories, reason, layout_plan)
        return build_generation_prompt(
            task["product_name"], task["rule_category"], slot_info,
            task["description"], task["specifications"], forced_variation,
            exclude_features, no_feature_available=no_feature_available,
            layout_plan=layout_plan, geometry_analysis=geometry_analysis,
                retry_feedback=feedback)

    # Try the new deterministic pipeline first for dimension slots — clean
    # product image + OpenCV corner detection + PIL-drawn arrows, with no
    # generative model ever touching the numbers/labels (see
    # try_deterministic_dimension_image). Falls back to the existing
    # model-drawn-arrows + verify/retry path, completely unchanged, on ANY
    # uncertainty (segmentation failure, ambiguous geometry, etc.) — this is
    # purely additive, never a risk to the proven fallback.
    used_deterministic_dimensions = False
    if is_dimension_slot:
        try:
            det_ok = try_deterministic_dimension_image(
                slot_reference_url, task["product_name"], task["rule_category"],
                task["description"], task["specifications"], layout_plan, dim_data,
                out_path, project_id, region, tokens)
        except Exception:
            det_ok = False
        if det_ok:
            used_deterministic_dimensions = True
            attempt_log.append({"attempt": 1, "valid": True,
                                "reason": "deterministic_opencv_pipeline_success",
                                "failure_categories": []})

    if used_deterministic_dimensions:
        result = {"status": "generated"}
    else:
        result = generate_image_with_verification(
            slot_reference_url, prompt, out_path, project_id, region, tokens, axis_labels,
            is_vehicle=is_vehicle, text_only_label=text_only_label,
            product_name=task["product_name"], description=task["description"],
            specifications=task["specifications"], image_type=slot_info["image_type"],
            forced_variation=forced_variation, exclude_features=exclude_features,
            retry_prompt_fn=retry_prompt_fn,
            attempt_log_fn=_log_attempt if is_dimension_slot else None,
            known_length_cm=known_length_cm)
    status = result["status"]
    generated = status == "generated" or status.startswith("generated_unverified")
    if is_dimension_slot and generated:
        # Stamp the disclaimer onto whatever file is about to be shipped or
        # reviewed — including a "generated_unverified" file, since a human
        # reviewer may still open it locally. Must happen AFTER
        # generate_image_with_verification so it never runs on an attempt
        # that was later discarded/regenerated, and before maybe_upload so
        # the uploaded copy already has it.
        try:
            add_size_disclaimer(out_path)
        except Exception:
            # Best-effort cosmetic step — never fail the whole slot over it.
            pass

    # A Lifestyle slot that exhausted every retry still stuck on a
    # SCALE_MISMATCH or PRODUCT_AUTHENTICITY verdict gets the same
    # "don't ship a known defect" treatment dimension slots get above (see
    # FAILURE_PRODUCT_AUTHENTICITY) — authenticity and real-world scale are
    # non-negotiable, unlike a cosmetic framing/composition miss on a
    # Lifestyle shot, which still ships as "Generated (needs review)" same
    # as before. A wrong-looking or oversized/undersized product in a
    # lifestyle scene is worse than a plain, correctly-scaled product photo
    # in its place. (is_dimension_slot is never true here — every dimension
    # slot returns earlier in this function, with its own equivalent
    # alt-angle-primary + secondary-chart handling above.)
    lifestyle_fallback_categories = {FAILURE_SCALE_MISMATCH, FAILURE_PRODUCT_AUTHENTICITY}
    needs_alt_fallback = (
        is_lifestyle_slot and status.startswith("generated_unverified")
        and lifestyle_fallback_categories.intersection(result.get("failure_categories", []))
    )

    bonus_row = None
    if needs_alt_fallback:
        # Exhausted MAX_GENERATION_ATTEMPTS (3) without ever passing
        # verification. Policy: a lifestyle scene with the wrong product or
        # the wrong real-world scale is worse than none. Keep the failed
        # attempt as a separate BONUS reference image (never counted as one
        # of the 6 required slots, never presented as verified) and replace
        # the actual required slot with a plain, easily-verifiable
        # alternate-angle photo instead, so the required 6 always ship
        # something legitimate.
        bonus_filename = filename.replace(".png", "_unverified_bonus.png")
        bonus_path = os.path.join(task["image_out_dir"], bonus_filename)
        shutil.copy(out_path, bonus_path)
        bonus_link, bonus_upload_failures = maybe_upload(
            uploader, upload_cache, bonus_filename, bonus_path, 0)
        bonus_row = {
            **row_base,
            "Slot": f"{slot_num}b",
            "Image_Type": f"{slot_info['image_type']} (unverified reference only)",
            "Status": "Generated (needs review)",
            "Image_Source": bonus_filename, "GCP_Link": bonus_link,
            "_bucket": "Generated", "_needs_review": True,
            "_upload_failed": bonus_upload_failures > 0,
        }

        # Deliberately a DIFFERENT image_type ("Additional Angle", not
        # "Lifestyle Image") passed into the prompt builder/verifier so it
        # doesn't re-trigger the lifestyle scale/authenticity check again —
        # this must read as a plain, easy-to-verify product photo, not a
        # repeat of whatever just failed. "description" must be overridden
        # too, not just "image_type" — slot_info['description'] (this
        # slot's ORIGINAL rule-master text, e.g. a lifestyle-scene
        # instruction) gets inserted verbatim as "Requirement: ..." by
        # build_generation_prompt regardless of image_type (confirmed real
        # defect on the equivalent dimension-slot code path — see above).
        alt_slot_info = {**slot_info, "image_type": "Additional Angle",
                         "description": ("Show a clean, different full or three-quarter "
                                         "view of the complete product on a plain "
                                         "background, distinct from this product's "
                                         "other images.")}
        alt_forced_variation = (
            "a clean, different full or three-quarter view of the complete "
            "product on a plain background — a plain studio product photo, "
            "NOT a real-world lifestyle scene"
        )
        alt_prompt = build_generation_prompt(
            task["product_name"], task["rule_category"], alt_slot_info,
            task["description"], task["specifications"],
            forced_variation=alt_forced_variation,
            low_quality_reference=low_quality_reference)
        result = generate_image_with_verification(
            slot_reference_url, alt_prompt, out_path, project_id, region, tokens,
            axis_labels=[], product_name=task["product_name"],
            description=task["description"], specifications=task["specifications"],
            image_type="Additional Angle")
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
        # is_dimension_slot is never true here — every dimension slot
        # returns earlier in this function (see above), so this is always
        # the ordinary generic "needs review" wording, never the strict
        # dimension one.
        display_status = "Generated (needs review)"
    else:
        display_status = status
    return {**row_base, "Status": display_status,
           "Image_Source": filename if generated else "", "GCP_Link": gcp_link,
           "_bucket": "Generated" if generated else "Failed",
           "_needs_review": status.startswith("generated_unverified"),
           "_upload_failed": upload_failed, "_bonus_row": bonus_row}


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
    feature_cache = FeatureExtractionCache()
    reference_cache = ReferenceAssessmentCache()
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
    live_counts = {"Existing": 0, "Generated": 0, "Reused": 0, "Failed": 0, "NeedsReview": 0,
                  "Skipped": 0}
    status_path = args.out + ".status.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def worker(task):
        if checkpoint.is_done(task["key"]):
            return checkpoint.get(task["key"])
        row = process_slot_task(task, project_id, region, tokens, uploader, upload_cache,
                                args.overwrite, feature_cache, reference_cache)
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
    # A dimension slot that exhausted all retries embeds a "_bonus_row" —
    # the failed dimension attempt, kept as a non-counted extra reference
    # line right after its real slot, never one of the 6 required rows (see
    # process_slot_task's alt-angle fallback).
    expanded_rows = []
    for row in final_rows:
        expanded_rows.append(row)
        if row.get("_bonus_row"):
            expanded_rows.append(row["_bonus_row"])
    final_rows = expanded_rows
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
          f"{counts['Reused']} reused from disk, {counts['Failed']} failed/unresolved, "
          f"{counts['Skipped']} skipped (no admin Size/Dimensions photo — not generated).")
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
                     "Generated (needs review)": "FFF3CD",
                     # By-design (no admin dimension photo -> not generated,
                     # see process_slot_task), not a failure — neutral gray,
                     # not the red FAILED_FILL used for real problem rows.
                     "Skipped (no admin dimension photo — not generated)": "E5E7EB"}
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
