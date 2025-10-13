import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import csv
from tqdm import tqdm
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel

# -----------------------------
# ตั้งค่า Tesseract ถ้าจำเป็น
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load processor and model
processor = TrOCRProcessor.from_pretrained('suchut/thaitrocr-base-handwritten-beta1')
model = VisionEncoderDecoderModel.from_pretrained('suchut/thaitrocr-base-handwritten-beta1')

# -----------------------------
# โหลดโมเดล Table Transformer (DL) สำหรับ detect ตาราง
table_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
table_model = DetrForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
device = "cuda" if torch.cuda.is_available() else "cpu"
table_model.to(device)

# -----------------------------
def preprocess_image(pil_img):
    """คืนค่า gray และ thresh สำหรับ OCR / detection fallback"""
    # PIL → OpenCV BGR
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ลด noise เบา ๆ
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Contrast enhancement โดย CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Unsharp mask (sharpen)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2)
    sharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    # Threshold แบบ adaptive
    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return sharp, thresh

def _to_pil_rgb(img_in):
    """แปลง numpy array หรือ PIL image เป็น PIL RGB สำหรับ DL detector"""
    if isinstance(img_in, Image.Image):
        return img_in.convert("RGB")
    arr = np.asarray(img_in)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 3:
        # assume BGR
        try:
            return Image.fromarray(arr[:, :, ::-1])
        except:
            return Image.fromarray(arr)
    return Image.fromarray(arr).convert("RGB")

def detect_table_dl(img_in, score_threshold=0.6, max_size=1600):
    """ตรวจจับตารางด้วย DL model, คืน list กล่อง [x1,y1,x2,y2]"""
    pil_img = _to_pil_rgb(img_in)
    orig_w, orig_h = pil_img.size
    scale = 1.0
    if max(orig_w, orig_h) > max_size:
        scale = max_size / float(max(orig_w, orig_h))
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_resized = pil_img.resize((new_w, new_h))
    else:
        pil_resized = pil_img

    inputs = table_processor(images=pil_resized, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = table_model(**inputs)

    target_sizes = torch.tensor([[pil_resized.size[1], pil_resized.size[0]]])
    if torch.cuda.is_available():
        target_sizes = target_sizes.to(device)

    results = table_processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]

    boxes = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        # map กลับถ้าปรับสเกล
        if scale != 1.0:
            inv = 1.0 / scale
            x1 *= inv; y1 *= inv; x2 *= inv; y2 *= inv
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def detect_table(thresh):
    """fallback detection แบบ morphology (ใช้ threshold จาก preprocessing)"""
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_mask = cv2.add(detect_horizontal, detect_vertical)
    return table_mask

def extract_cells(img, table_mask):
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cell_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:
            cell = img[y:y+h, x:x+w]
            cell_images.append((x, y, cell))
    cell_images = sorted(cell_images, key=lambda x: (x[1], x[0]))
    return cell_images

def ocr_cells2(cell_images):
    results = []
    for (x, y, cell_img) in cell_images:
        # แปลงเป็น grayscale ถ้าจำเป็น
        if len(cell_img.shape) == 3 and cell_img.shape[2] == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img

        # เตรียม cell สำหรับ OCR: ขยาย + เบลอเล็กน้อย
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # OCR พิมพ์ไทย + อังกฤษ
        text = pytesseract.image_to_string(
            gray, lang="tha+eng",
            config="--psm 6 --oem 3 -c preserve_interword_spaces=1"
        ).strip()

        results.append((x, y, text))
    return results

def ocr_cells(cell_images):
    results = []
    for (x, y, cell_img) in cell_images:
        # ตรวจสอบจำนวน channel ก่อน
        if len(cell_img.shape) == 3 and cell_img.shape[2] == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img  # เป็น grayscale อยู่แล้ว

        # แปลงกลับเป็น PIL.Image
        pil_img = Image.fromarray(gray).convert("RGB")

        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # # OCR ด้วยภาษาไทย + อังกฤษ
        # text = pytesseract.image_to_string(
        #     gray, lang="tha+eng", config="--psm 6"
        # ).strip()

        results.append((x, y, generated_text))
    return results

def group_to_table(results, row_tolerance=15):
    rows = []
    current = []
    last_y = None
    for (x, y, text) in results:
        if last_y is None:
            last_y = y
        if abs(y - last_y) > row_tolerance:
            rows.append(current)
            current = []
            last_y = y
        current.append(text)
    if current:
        rows.append(current)
    return rows

def save_to_csv(all_rows, out_path="output.csv"):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for rows in all_rows:
            writer.writerows(rows)
            writer.writerow([])

def extract_tables_from_pdf(pdf_path, out_csv="output.csv"):
    print(f"📘 Processing PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=300)
    all_rows = []

    for i, page in enumerate(tqdm(pages, desc="Pages")):
        sharp, thresh = preprocess_image(page)

        # ใช้ DL detect ตาราง
        boxes = detect_table_dl(page)
        if not boxes:
            # fallback morphological detection
            mask = detect_table(thresh)
            cell_images = extract_cells(sharp, mask)
        else:
            cell_images = []
            page_arr = np.array(page.convert("RGB"))
            for (x1, y1, x2, y2) in boxes:
                # crop region from color page
                crop = page_arr[y1:y2, x1:x2]
                # crop grayscale version for cell detection
                crop_gray = cv2.cvtColor(crop[:, :, ::-1], cv2.COLOR_BGR2GRAY)
                mask = detect_table(crop_gray)
                cells = extract_cells(crop, mask)
                # adjust cell x,y relative to page
                for (cx, cy, cell_img) in cells:
                    cell_images.append((x1 + cx, y1 + cy, cell_img))

        if not cell_images:
            # ถ้าไม่มี cell ตรวจเจอ → OCR ทั้งหน้า
            text = pytesseract.image_to_string(
                sharp, lang="tha+eng",
                config="--psm 3 --oem 3"
            )
            all_rows.append([[text.strip()]])
        else:
            ocr_res = ocr_cells(cell_images)
            rows = group_to_table(ocr_res)
            if rows:
                all_rows.append(rows)

    save_to_csv(all_rows, out_csv)
    print(f"✅ Done! Output saved to {out_csv}")

if __name__ == "__main__":
    pdf_file = "test.pdf"
    extract_tables_from_pdf(pdf_file, "tables_output.csv")
