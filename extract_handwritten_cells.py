import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

# === CONFIG ===
PDF_PATH = "test.pdf"
OUTPUT_DIR = "output"
DPI = 300

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# === STEP 1: Convert PDF to images ===
print("Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=DPI)

def detect_tables_and_extract_cells(image, page_index):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    # --- Find table structure using lines ---
    horizontal = thresh.copy()
    vertical = thresh.copy()

    # horizontal lines
    horizontalsize = int(horizontal.shape[1] / 20)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # vertical lines
    verticalsize = int(vertical.shape[0] / 20)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # Combine
    mask = horizontal + vertical
    joints = cv2.bitwise_and(horizontal, vertical)

    # --- Find contours of cells ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_idx = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 30:  # skip small boxes
            continue

        cell = gray[y:y+h, x:x+w]
        if is_handwritten(cell):
            img_name = f"page{page_index:02d}_cell{cell_idx:04d}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", img_name), cell)

            label_path = os.path.join(OUTPUT_DIR, "labels", img_name.replace(".png", ".txt"))
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("")  # empty for manual labeling
            cell_idx += 1

def is_handwritten(cell):
    """Heuristic handwriting detection using texture and variance."""
    if np.mean(cell) > 230:
        return False  # too white (empty cell)

    # Laplacian variance: high variance → likely handwriting or textured
    lap_var = cv2.Laplacian(cell, cv2.CV_64F).var()
    return lap_var > 150  # tweak threshold (150-250) depending on document

# === STEP 2: Process each page ===
print("Detecting tables and extracting handwritten cells...")
for i, page in enumerate(tqdm(pages)):
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    detect_tables_and_extract_cells(img, i)

print(f"✅ Extraction complete. Check '{OUTPUT_DIR}/images' and '{OUTPUT_DIR}/labels'")
