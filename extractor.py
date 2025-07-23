# [optional] สำหรับ Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"

# แปลง PDF เป็น list ของภาพ
file_path = "62 สำนักงานเขตสายไหม.pdf"
file_path2 = "test.pdf"
# images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)  #, poppler_path=poppler_path

# # OCR ทีละหน้า
# text_all = ""
# for i, img in enumerate(images):
#     text = pytesseract.image_to_string(img, lang="eng+tha")
#     print(f"--- Page {i+1} ---")
#     print(text[:500])  # แสดงบางส่วน
#     text_all += f"\n\n--- Page {i+1} ---\n{text}"

# # บันทึกข้อความทั้งหมดลงไฟล์
# with open("output.txt", "w", encoding="utf-8") as f:
#     f.write(text_all)

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from multiprocessing import Pool, cpu_count

# สำหรับ Windows ถ้าจำเป็น
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_page(args):
    pdf_path, page_num, lang = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)

    text = page.get_text("text")
    print(f"--- Page {page_num+1} ---")
    if text.strip():
        print("text layer")
        return (page_num, f"--- Page {page_num+1} ---\n{text.strip()}")
    else:
        print("OCR")
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img, lang=lang)
        return (page_num, f"--- Page {page_num+1} (OCR) ---\n{ocr_text.strip()}")

def extract_pdf_parallel(pdf_path, output_txt, lang="tha+eng"):
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()  # ปิดทันที เพราะแต่ละ process จะเปิดเอง

    tasks = [ (pdf_path, i, lang) for i in range(total_pages) ]

    # ใช้ CPU cores เท่าที่มี หรือ 4
    with Pool(processes=min(cpu_count(), 4)) as pool:
        results = pool.map(process_page, tasks)

    # เรียงผลลัพธ์ตามลำดับหน้า
    results.sort(key=lambda x: x[0])

    # รวมข้อความทุกหน้า
    all_text = "\n\n".join([r[1] for r in results])

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"\n🎉 Done. Output saved to: {output_txt}")

if __name__ == "__main__":
    extract_pdf_parallel(file_path, "output_budget.txt")

