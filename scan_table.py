import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import csv
from tqdm import tqdm
from PIL import Image

# -----------------------------
# หาก Tesseract ไม่ได้อยู่ใน PATH ให้ระบุ path ตรงนี้
# เช่น:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # เพิ่ม contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    return img, thresh

# -----------------------------
def detect_table(thresh):
    # เส้นแนวนอน
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # เส้นแนวตั้ง
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # รวมเส้น
    table_mask = cv2.add(detect_horizontal, detect_vertical)
    return table_mask

# -----------------------------
def extract_cells(img, table_mask):
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cell_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:  # ตัด noise เล็กๆทิ้ง
            cell_img = img[y:y+h, x:x+w]
            cell_images.append((x, y, cell_img))

    # sort: row → col
    cell_images = sorted(cell_images, key=lambda x: (x[1], x[0]))
    return cell_images

# -----------------------------
def ocr_cells(cell_images):
    results = []
    for (x, y, cell_img) in cell_images:
        # แปลงเป็น grayscale สำหรับ Tesseract
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        # OCR ด้วยภาษาไทย + อังกฤษ
        text = pytesseract.image_to_string(
            gray, lang="tha+eng", config="--psm 6"
        ).strip()

        results.append((x, y, text))
    return results

# -----------------------------
def group_to_table(results, row_tolerance=15):
    rows = []
    current_row = []
    last_y = None

    for (x, y, text) in results:
        if last_y is None:
            last_y = y
        if abs(y - last_y) > row_tolerance:  # ขึ้นบรรทัดใหม่
            rows.append(current_row)
            current_row = []
            last_y = y
        current_row.append(text)
    if current_row:
        rows.append(current_row)
    return rows

# -----------------------------
def save_to_csv(all_rows, out_path="output.csv"):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for rows in all_rows:
            writer.writerows(rows)
            writer.writerow([])  # เว้นบรรทัดคั่นหน้าต่างๆ

# -----------------------------
def extract_tables_from_pdf(pdf_path, out_csv="output.csv"):
    print(f"📖 Processing PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=300)

    all_rows = []
    for i, page in enumerate(tqdm(pages, desc="Processing pages")):
        img, thresh = preprocess_image(page)
        table_mask = detect_table(thresh)
        cell_images = extract_cells(img, table_mask)

        if not cell_images:
            continue

        ocr_results = ocr_cells(cell_images)
        rows = group_to_table(ocr_results)
        if rows:
            all_rows.append(rows)

    save_to_csv(all_rows, out_csv)
    print(f"✅ Done! Extracted tables saved to {out_csv}")

# -----------------------------
if __name__ == "__main__":
    pdf_file = "test.pdf"   # 👉 ใส่ไฟล์ PDF ของคุณ
    extract_tables_from_pdf(pdf_file, "tables_output.csv")
