"""
Creates demo data so you can test the classify + generate steps immediately,
without needing your real OZI_API_KEY yet.

It draws one simple synthetic "front view" toy photo locally (a soft toy
placeholder, plain white background) and writes a products_detail_demo.csv
in the exact format fetch_product_details.py would normally produce.

Run: python3 create_demo_data.py
Then:
    python3 ../scripts/classify_images.py --input products_detail_demo.csv \
        --rules ../rules/toys_rule_master.json --out classification_result_demo.csv
    python3 ../scripts/generate_missing_images_gcp.py --products products_detail_demo.csv \
        --classification classification_result_demo.csv --rules ../rules/toys_rule_master.json \
        --image_out_dir generated_images_demo --out final_output_demo.xlsx
"""
import os
import csv
from PIL import Image, ImageDraw

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(OUT_DIR, "demo_soft_toy_front.png")
CSV_PATH = os.path.join(OUT_DIR, "products_detail_demo.csv")


def draw_demo_toy_image(path: str):
    """A simple synthetic placeholder — a teddy-bear-like silhouette on
    white, just so the pipeline has something to classify/edit. Swap this
    for a real product photo path once you're testing with real data."""
    img = Image.new("RGB", (800, 800), "white")
    d = ImageDraw.Draw(img)
    brown = (150, 105, 65)
    # head
    d.ellipse([300, 120, 500, 320], fill=brown)
    # ears
    d.ellipse([270, 90, 350, 170], fill=brown)
    d.ellipse([450, 90, 530, 170], fill=brown)
    # body
    d.ellipse([250, 300, 550, 650], fill=brown)
    # arms
    d.ellipse([180, 340, 280, 520], fill=brown)
    d.ellipse([520, 340, 620, 520], fill=brown)
    # eyes + nose
    d.ellipse([355, 190, 375, 210], fill="black")
    d.ellipse([425, 190, 445, 210], fill="black")
    d.ellipse([385, 220, 415, 245], fill=(90, 60, 40))
    img.save(path)


def write_demo_csv(path: str, image_path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Product_ID", "SKU", "Name", "Category_L1", "Category_L2", "Category_L3",
            "Description", "Image_Count", "Image_URLs", "Fetch_Status",
        ])
        writer.writerow([
            999001, "DEMO0001SKU", "Demo Brown Teddy Bear Soft Toy, 30cm",
            "Soft Toys", "", "",
            "Soft plush teddy bear, brown, suitable for ages 3+.",
            1, image_path, "ok",
        ])


if __name__ == "__main__":
    draw_demo_toy_image(IMG_PATH)
    write_demo_csv(CSV_PATH, IMG_PATH)
    print(f"Demo image  -> {IMG_PATH}")
    print(f"Demo input  -> {CSV_PATH}")
    print("\nThis demo product has only 1 image and is mapped to category "
          "'Soft Toys', which needs 6 slots -> the classifier should find "
          "slot 1 (front) covered and 5 slots missing, ready for generation.")
