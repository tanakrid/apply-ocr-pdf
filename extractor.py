import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import re
from tqdm import tqdm
import time
import camelot
import pandas as pd

# สำหรับ Windows ถ้าจำเป็น
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"

file_path = "62 สำนักงานเขตสายไหม.pdf"
file_path2 = "test.pdf"

# 🔷 เช็คว่าเป็นขยะหรือไม่
def looks_gibberish(text):
    clean = re.sub(r'[ก-๙a-zA-Z0-9\s.,:;()\-]', '', text)
    ratio = len(clean) / max(len(text),1)
    if ratio > 0.3:
        return True
    if len(text.strip()) < 10:
        return True
    return False

# 🩷 ตรวจว่า csv มีคุณภาพโอเคไหม
def validate_table(df, min_cols=2, max_empty_ratio=0.5):
    if df.shape[1] < min_cols:
        return False
    empty_ratio = (df.isnull() | (df=='')).mean().mean()
    if empty_ratio > max_empty_ratio:
        return False
    col_counts = df.apply(lambda r: r.count(), axis=1)
    if col_counts.std() > 2:
        return False
    return True

# สำหรับ text layer
def save_table_as_csv(pdf_path, page_num, output_folder):
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num+1))
        for idx, table in enumerate(tables):
            # เรียก validate และแจ้งผล
            if validate_table(table.df):
                out_csv = os.path.join(output_folder, f"page_{page_num+1}_table_{idx+1}.csv")
                table.to_csv(out_csv)
        if tables:
            return True
    except Exception as e:
        print(f"⚠️ Table extraction error on page {page_num+1}: {e}")
    return False

def extract_table_as_string(pdf_path, page_num, flavor="lattice"):
    csv_strings = []
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num+1), flavor=flavor)
        for idx, table in enumerate(tables):
            csv_str = table.df.to_csv(index=False)
            # เรียก validate และแจ้งผล
            if validate_table(table.df):
                csv_strings.append(f"--- Table {idx+1} (CSV) ---\n{csv_str}")
    except Exception as e:
        print(f"⚠️ Table extraction error on page {page_num+1}: {e}")
    return "\n\n".join(csv_strings)

def auto_rotate_image(pil_img):
    try:
        pil_img.info["dpi"] = (300, 300)  # ✅ เพิ่ม DPI ให้ภาพ
        osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        angle = osd.get('rotate', 0)
        if angle != 0:
            pil_img = pil_img.rotate(-angle, expand=True)
    except pytesseract.TesseractError as e:
        print(f"⚠️ Warning: OSD rotation failed - {e}. Skipping rotation.")
    return pil_img


# --------- OCR ตารางจากภาพสแกน ---------
def extract_table_from_image(img, lang='tha+eng', psm=6):
    # input: np.ndarray (BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    horizontal = thresh.copy()
    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical = thresh.copy()
    rows = vertical.shape[0]
    vertical_size = rows // 20
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    mask = horizontal + vertical

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:
            cells.append((x, y, w, h))

    if not cells or len(cells) < 4:
        return None  # อาจไม่ใช่ตาราง

    cells = sorted(cells, key=lambda b: (b[1]//10, b[0]))
    rows_list = []
    current_row = []
    last_y = -1
    for box in cells:
        x, y, w, h = box
        if last_y == -1 or abs(y - last_y) <= 10:
            current_row.append(box)
            last_y = y
        else:
            rows_list.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            last_y = y
    if current_row:
        rows_list.append(sorted(current_row, key=lambda b: b[0]))

    table_data = []
    for row in rows_list:
        row_data = []
        for cell in row:
            x, y, w, h = cell
            cell_img = img[y:y+h, x:x+w]
            pil_img = Image.fromarray(cell_img)
            config = f'--psm {psm}'
            text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
            text = text.strip().replace('\n', ' ')
            row_data.append(text)
        table_data.append(row_data)

    df = pd.DataFrame(table_data)

    # เรียก validate และแจ้งผล
    if validate_table(df):
        return df.to_csv(index=False, header=False)
    else:
        return None

def preprocess_image(pix):
    # แปลง Pixmap ของ fitz เป็น OpenCV image
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    # sharpen
    sharpen = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0,0), 3), -0.5, 0)
    _, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def process_page(args):
    pdf_path, page_num, lang, psm = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)

    result = ""
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (b[1], b[0]))  # sort ตาม y, x
    lines = [" ".join(b[4].split()) for b in blocks]
    text = "\n".join(lines)
    # print(f"--- Page {page_num+1} ---")
    if not looks_gibberish(text):
        # print("text layer")
        result = f"--- Page {page_num+1} ---\n{text}"

        # ถ้ามีตาราง → แปะ string ของตารางต่อท้าย
        table_str = extract_table_as_string(pdf_path, page_num)
        if table_str:
            result += "\n\n" + table_str
    else:
        # print("OCR")
        pix = page.get_pixmap(dpi=300)
        img = preprocess_image(pix)
        pil_img = Image.fromarray(img)
        pil_img = auto_rotate_image(pil_img)
        config = f'--psm {psm}'
        ocr_text = pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()
        result = f"--- Page {page_num+1} (OCR) ---\n{ocr_text.strip()}"

        table_img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # PIL → BGR
        table_str = extract_table_from_image(table_img, lang, psm)
        if table_str:
            result += f"\n\n--- Table Detected (CSV) ---\n{table_str}"

    return (page_num, result)

def extract_pdf_parallel(pdf_path, output_txt, lang="tha+eng", psm=6):
    start_time = time.time()
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()  # ปิดทันที เพราะแต่ละ process จะเปิดเอง

    tasks = [ (pdf_path, i, lang, psm) for i in range(total_pages) ]

    # ใช้ CPU cores เท่าที่มี หรือ 4
    results = []
    with Pool(processes=min(cpu_count(), 4)) as pool:
        for res in tqdm(pool.imap_unordered(process_page, tasks), total=total_pages, desc="Processing pages"):
            results.append(res)

    # เรียงผลลัพธ์ตามลำดับหน้า
    results.sort(key=lambda x: x[0])

    # รวมข้อความทุกหน้า
    all_text = "\n\n".join([r[1] for r in results])

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(all_text)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"\n🎉 Done. Output saved to: {output_txt}")
    print(f"⏱️ Elapsed time: {int(mins)}m {int(secs)}s")

if __name__ == "__main__":
    extract_pdf_parallel(file_path, "output_budget_3.txt")

