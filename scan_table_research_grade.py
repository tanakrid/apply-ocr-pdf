import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import csv
import torch
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from paddleocr import PaddleOCR

# -----------------------------
# โหลดโมเดลสำหรับ Table Detection และ OCR
table_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
table_model = DetrForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

thai_processor = TrOCRProcessor.from_pretrained("suchut/thaitrocr-base-handwritten-beta1")
thai_model = VisionEncoderDecoderModel.from_pretrained("suchut/thaitrocr-base-handwritten-beta1")

# paddle_ocr = PaddleOCR(lang='th', use_angle_cls=True, show_log=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
thai_model.to(device)

# -----------------------------
def preprocess_image1(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ใช้ CLAHE เพิ่ม contrast โดยไม่ทำลายเส้นลายมือ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ลด noise แต่เก็บขอบไว้
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray

def preprocess_image2(pil_img):
    # แปลงจาก PIL -> OpenCV (RGB -> BGR)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # ลด noise เล็กน้อย
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ลด noise แต่เก็บขอบไว้
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # เพิ่ม contrast เพื่อช่วย OCR
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # ทำ adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10
    )

    return gray, thresh

def _to_pil_rgb(img_in):
    """รับ PIL.Image หรือ numpy array (H,W) หรือ (H,W,3) และคืน PIL RGB"""
    if isinstance(img_in, Image.Image):
        return img_in.convert("RGB")
    arr = np.asarray(img_in)
    if arr.ndim == 2:
        # grayscale -> convert to RGB
        return Image.fromarray(arr).convert("RGB")
    if arr.ndim == 3:
        # detect channel order: if likely RGB or BGR
        if arr.shape[2] == 3:
            # assume it's BGR if dtype is uint8 and likely from cv2
            # convert BGR -> RGB
            try:
                # Heuristic: if mean of first channel similar to others, still safe
                return Image.fromarray(arr[:, :, ::-1])  # BGR->RGB
            except Exception:
                return Image.fromarray(arr)
    # fallback
    return Image.fromarray(arr).convert("RGB")

def detect_table_dl2(img_in, score_threshold=0.6, max_size=1600):
    """
    img_in: PIL.Image or numpy array (grayscale or BGR)
    คืน: list of boxes [x1,y1,x2,y2] ในพิกัดของรูปต้นฉบับ
    """
    pil_img = _to_pil_rgb(img_in)

    # ลดขนาดถ้ารูปใหญ่เกิน (เก็บ scale เพื่อ map กลับ)
    orig_w, orig_h = pil_img.size
    scale = 1.0
    if max(orig_w, orig_h) > max_size:
        scale = max_size / float(max(orig_w, orig_h))
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_resized = pil_img.resize((new_w, new_h))
    else:
        pil_resized = pil_img

    # prepare inputs
    inputs = table_processor(images=pil_resized, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k:v.cuda() for k,v in inputs.items()}

    with torch.no_grad():
        outputs = table_model(**inputs)

    # target_sizes expects (H, W)
    target_sizes = torch.tensor([[pil_resized.size[1], pil_resized.size[0]]])
    if torch.cuda.is_available():
        target_sizes = target_sizes.cuda()

    results = table_processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]

    boxes = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        # label filtering: depends on model; adjust if needed
        # append scaled boxes back to original image coords
        x1, y1, x2, y2 = box.tolist()
        # scale back to original size if resized
        if scale != 1.0:
            inv_scale = 1.0 / scale
            x1 *= inv_scale; y1 *= inv_scale; x2 *= inv_scale; y2 *= inv_scale
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

# -----------------------------
def detect_table_dl(pil_img):
    """ใช้ Table Transformer ตรวจหาตาราง"""
    inputs = table_processor(images=pil_img, return_tensors="pt")
    outputs = table_model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = table_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

    boxes = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if label == 0:  # table only
            boxes.append(box.tolist())
    return boxes

# -----------------------------
def classify_handwriting(cell_img):
    """ใช้ heuristic ว่า cell เป็นลายมือหรือไม่"""
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    variance = np.var(gray)
    return variance > 500  # ค่า variance สูงบ่งบอกว่ามีเส้นโค้งและลายมือ

# -----------------------------
def ocr_cell(cell_img, handwriting=False):
    """OCR แต่ละ cell"""
    if handwriting:
        pil_img = Image.fromarray(cell_img).convert("RGB")
        pixel_values = thai_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        generated_ids = thai_model.generate(pixel_values)
        text = thai_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    # else:
    #     result = paddle_ocr.ocr(cell_img, cls=True)
    #     return " ".join([line[1][0] for line in result[0]]) if result and result[0] else ""

# -----------------------------
def extract_tables_from_pdf(pdf_path, out_csv="output.csv"):
    print(f"📖 Processing PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=300)
    all_rows = []

    for i, page in enumerate(tqdm(pages, desc="Processing pages")):
        gray, thresh = preprocess_image2(page)

        # ตรวจหาตารางด้วย DL
        table_boxes = detect_table_dl(page)
        print("table_boxes:", table_boxes)
        if not table_boxes:
            continue

        for box in table_boxes:
            x1, y1, x2, y2 = map(int, box)
            table_crop = np.array(page)[y1:y2, x1:x2]

            # ตัดเป็นกริดเบื้องต้น (โดยประมาณ)
            h, w = table_crop.shape[:2]
            cell_h = h // 10
            cell_w = w // 5
            rows = []
            for r in range(0, h, cell_h):
                row_texts = []
                for c in range(0, w, cell_w):
                    cell = table_crop[r:r+cell_h, c:c+cell_w]
                    handwriting = classify_handwriting(cell)
                    text = ocr_cell(cell, handwriting)
                    row_texts.append(text)
                rows.append(row_texts)
            all_rows.extend(rows)

    # บันทึก CSV
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row)

    print(f"✅ Done! Extracted tables saved to {out_csv}")

# -----------------------------
if __name__ == "__main__":
    pdf_file = "test.pdf"
    extract_tables_from_pdf(pdf_file, "tables_output.csv")
