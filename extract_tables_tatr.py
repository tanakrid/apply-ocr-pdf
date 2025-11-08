import os
import cv2
import torch
import datetime
import numpy as np
from PIL import Image
from transformers import AutoProcessor, TableTransformerForObjectDetection
from pdf2image import convert_from_path


# ======= CONFIG =======
PDF_PATH = "test.pdf"
OUTPUT_EXTRACTED_CELLS_DIR = os.path.join("outputs", "extracted_cells")
OUTPUT_DIR = str(os.path.join(OUTPUT_EXTRACTED_CELLS_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
DEBUG_VISUALIZE = True
DEBUG_BORDER_SIZE = 5
MIN_TABLE_CONFIDENCE = 0.8
# ======================


print("📦 Loading Table Transformer (microsoft/table-transformer-detection)...")
processor = AutoProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(p.convert("RGB")) for p in pages]


def detect_tables_with_tatr(image):
    pil_img = Image.fromarray(image)
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_object_detection(
        outputs, threshold=MIN_TABLE_CONFIDENCE, target_sizes=[pil_img.size[::-1]]
    )[0]

    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        if label_name == "table":
            tables.append((box.int().tolist(), float(score)))
    return tables


def extract_table_regions(image, tables, page_idx):
    crops = []
    for i, (box, score) in enumerate(tables):
        x1, y1, x2, y2 = box
        # ปรับ margin (จำนวน pixel รอบขอบ)
        margin = 10

        # ตรวจสอบไม่ให้เกินขนาดภาพ
        height, width = image.shape[:2]

        # คำนวณขอบใหม่โดยไม่ให้เกินขอบภาพ
        x1_m = max(x1 - margin, 0)
        y1_m = max(y1 - margin, 0)
        x2_m = min(x2 + margin, width)
        y2_m = min(y2 + margin, height)

        # crop table พร้อม margin
        table_crop = image[y1_m:y2_m, x1_m:x2_m]
        crops.append((x1, y1, x2, y2, table_crop))

        if DEBUG_VISUALIZE:
            cv2.rectangle(image, (x1_m, y1_m), (x2_m, y2_m), (0, 200, 0), DEBUG_BORDER_SIZE)
            cv2.putText(image, f"Table {i+1} ({score:.2f})", (x1_m, y1_m - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), DEBUG_BORDER_SIZE)

    if DEBUG_VISUALIZE:
        vis_path = os.path.join(OUTPUT_DIR, f"page_{page_idx+1}_detected.jpg")
        cv2.imwrite(vis_path, image)

    return crops


def extract_cells_from_table(np_crop, base_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    gray = cv2.cvtColor(np_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binary invert เพื่อเห็นช่องว่างและลายมือได้ดีขึ้น
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 🔧 (2) Morphological closing เพื่อเชื่อมเส้นตารางที่ขาด
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 🔧 (1) ค้นหาเส้นตารางทั้งแนวตั้งและแนวนอน
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detect_h = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_h)
    detect_v = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_v)
    table_mask = cv2.add(detect_h, detect_v)

    # 🔧 (1) ใช้ RETR_TREE เพื่อดึง cell ทุกระดับ (รวม inner cell)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cell_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # 🔧 (3) ไม่ตัด cell ออกเร็วเกินไป — ปรับเงื่อนไขกรองให้กว้างขึ้น
        if w < 20 or h < 15 or area < 300:
            continue
        cell_boxes.append((x, y, w, h))

    if not cell_boxes:
        print(f"⚠️ No cells found in {base_name}")
        return

    # เรียงลำดับ cell ตามตำแหน่ง
    cell_boxes = sorted(cell_boxes, key=lambda c: (c[1] // 20, c[0]))

    # Group เป็นแถว
    rows = []
    current_y = None
    current_row = []
    for box in cell_boxes:
        x, y, w, h = box
        if current_y is None:
            current_y = y
        if abs(y - current_y) < 15:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = y
    if current_row:
        rows.append(current_row)

    # Extract cell per row
    for ridx, row in enumerate(rows):
        row_dir = os.path.join(save_dir, f"row_{ridx+1}")
        os.makedirs(row_dir, exist_ok=True)

        row_imgs = []
        max_h = max(h for (_, _, _, h) in row)

        for cidx, (x, y, w, h) in enumerate(row):
            cell = np_crop[y:y+h, x:x+w]

            # Resize ให้สูงเท่ากันก่อน concatenate
            cell_resized = cv2.resize(cell, (w, max_h))

            # 🔧 (3) ข้าม cell ว่างจริง ๆ เท่านั้น
            gray_cell = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY)
            if np.mean(gray_cell) > 250 and np.std(gray_cell) < 2:
                continue  # ข้าม cell ว่างจริง ๆ

            out_path = os.path.join(row_dir, f"{base_name}_r{ridx+1}_c{cidx+1}.jpg")
            cv2.imwrite(out_path, cell_resized)
            row_imgs.append(cell_resized)

        # ถ้าแถวทั้งแถวว่าง ไม่บันทึก
        if len(row_imgs) == 0:
            continue

        if DEBUG_VISUALIZE:
            vis = np_crop.copy()
            for (x, y, w, h) in row:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), DEBUG_BORDER_SIZE)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_row{ridx+1}_preview.jpg"), vis)


def extract_tables_from_pdf(pdf_path, output_dir):
    pages = pdf_to_images(pdf_path)

    for page_idx, page_img in enumerate(pages):
        print(f"🔍 Processing page {page_idx+1}/{len(pages)}...")
        tables = detect_tables_with_tatr(page_img)
        if not tables:
            print("⚠️ No tables detected on this page.")
            continue

        crops = extract_table_regions(page_img, tables, page_idx)
        for t_idx, (_, _, _, _, table_crop) in enumerate(crops):
            base_name = f"page{page_idx+1}_table{t_idx+1}"
            table_dir = os.path.join(output_dir, base_name)
            extract_cells_from_table(table_crop, base_name, table_dir)

    print("✅ Cell extraction completed.")


if __name__ == "__main__":
    extract_tables_from_pdf(PDF_PATH, OUTPUT_DIR)
