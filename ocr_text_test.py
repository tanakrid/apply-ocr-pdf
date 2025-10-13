import os
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel

# Optional: ตั้งค่า path ของ Tesseract (ถ้าจำเป็น)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(pil_img):
    """ปรับ contrast และลด noise เบา ๆ"""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    return gray


def ocr_image(gray):
    """OCR ด้วยภาษาไทย + อังกฤษ"""
    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(gray, lang="tha+eng", config=config)
    return text

# Load processor and model
processor = TrOCRProcessor.from_pretrained('suchut/thaitrocr-base-handwritten-beta1')
model = VisionEncoderDecoderModel.from_pretrained('suchut/thaitrocr-base-handwritten-beta1')

def ocr(cell_image):
    # ตรวจสอบจำนวน channel ก่อน
    if len(cell_image.shape) == 3 and cell_image.shape[2] == 3:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_image  # เป็น grayscale อยู่แล้ว

    # แปลงกลับเป็น PIL.Image
    pil_img = Image.fromarray(gray).convert("RGB")

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def extract_text_from_pdf(pdf_path, out_txt="ocr_output.txt"):
    print(f"📖 Processing {pdf_path}...")
    pages = convert_from_path(pdf_path, dpi=250)

    all_text = ""
    for i, page in enumerate(tqdm(pages, desc="Running OCR")):
        gray = preprocess_image(page)
        # text = ocr_image(gray)
        text = ocr(gray)
        all_text += f"\n\n===== PAGE {i+1} =====\n\n{text.strip()}"

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"✅ OCR completed. Text saved to {out_txt}")


if __name__ == "__main__":
    pdf_file = "test.pdf"   # 👉 เปลี่ยนเป็นชื่อไฟล์ PDF ที่ต้องการทดสอบ
    extract_text_from_pdf(pdf_file)
