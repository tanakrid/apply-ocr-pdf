## APPLY-OCR-PDF

โปรเจคนี้เป็นชุดสคริปต์ Python สำหรับประมวลผลเอกสาร (PDF / รูปภาพ) เพื่อสกัดตารางและเซลล์ภายในตารางแล้วส่งออกผลลัพธ์ในรูปแบบภาพและไฟล์ข้อมูล (เช่น CSV/ภาพ/labels) โดยมุ่งเน้นการใช้งานกับชุดข้อมูลสแกนฟอร์มภาษาไทย

### ไฮไลท์
- หลายเวอร์ชันของตัวดึงตาราง (ตระกูล `extract_tables_tatr*`) ที่ทดลองเทคนิคต่าง ๆ
- สคริปต์ตรวจจับเซลล์ในตารางและสกัดข้อความจากเซลล์
- ตัวอย่างการตั้งค่า environment และไฟล์ `requirements.txt` เพื่อใช้งาน

## โครงสร้างโปรเจค (สำคัญ)
- ไฟล์สคริปต์หลัก: `extract_tables_tatr.py`, `extract_tables_tatr_v5.py`, `extract_tables_tatr_v5_1.py`, `extract_table_cells_accurate.py`, `extract_handwritten_cells.py`, `extract_form_tables_advanced.py` และอื่น ๆ
- ตัวช่วย/ทดสอบ: `test.py`, `test_use_new_ocr.py`, `ocr_text_test.py`
- ข้อมูลตัวอย่างและเทรนนิง: โฟลเดอร์ `dataset/`
- เอาต์พุตตัวอย่าง: โฟลเดอร์ `output/`, `output_cells_*`, `example_output_cells_extract_tables_tatr_v5_1/`
- ไฟล์ dependencies: `requirements.txt`

## ข้อกำหนด (Requirements)
- Python 3.8+ (แนะนำ 3.8 — 3.11)
- ติดตั้งแพ็กเกจจาก `requirements.txt`

## การติดตั้ง (Windows - cmd.exe)
เปิด command prompt และรันคำสั่งต่อไปนี้ (ตัวอย่างสำหรับการตั้ง virtualenv แบบง่าย):

```cmd
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

หมายเหตุ: ถ้าคุณใช้ conda หรือเครื่องมือจัดการ environment อื่น ๆ ให้ปรับคำสั่งตามเครื่องมือที่ใช้

## การใช้งานตัวอย่าง (ตัวอย่างทั่วไป)
โปรเจคนี้ประกอบด้วยหลายสคริปต์ที่อาจรับพารามิเตอร์ต่างกัน ดังนั้นคำสั่งตัวอย่างด้านล่างเป็นรูปแบบทั่วไป — ตรวจสอบ header หรือ docstring ของสคริปต์แต่ละไฟล์เพื่อดูพารามิเตอร์จริง

```cmd
# ประมวลผลไฟล์ PDF/รูปภาพ เพื่อสกัดตาราง
python extract_tables_tatr_v5.py --input "path\to\input.pdf" --output "path\to\output_dir"

# สกัดเซลล์อย่างละเอียด
python extract_table_cells_accurate.py --input "path\to\image_or_page" --out_dir "output_cells_1"

# สคริปต์ทดสอบ OCR / ตรวจสอบผล
python test_use_new_ocr.py --input "path\to\sample.png"
```

ถ้าสคริปต์ที่คุณเรียกไม่มีพารามิเตอร์แบบ CLI ให้เปิดไฟล์เพื่อแก้ไขค่า `INPUT/OUTPUT` ภายในโค้ดหรือปรับให้รับพารามิเตอร์ตามต้องการ

## รูปแบบเอาต์พุต
- โฟลเดอร์ `output/` และ `output_cells_*` จะเก็บภาพที่สกัดออกมา, ไฟล์ label และผลระหว่างทาง
- ตาราง/เซลล์มักถูกเซฟเป็นโฟลเดอร์ย่อยตามหน้าและหมายเลขตาราง เช่น `page1_table1/`
- มีตัวอย่างเอาต์พุตอยู่ใน `example_output_cells_extract_tables_tatr_v5_1/` เพื่อใช้เป็นการอ้างอิง

## การปรับแต่งและพัฒนา
- หากต้องการปรับพารามิเตอร์การตรวจจับขอบหรือ threshold ดูตัวแปรที่เกี่ยวข้องในสคริปต์ที่ใช้งาน (เช่น ค่าการ blur, ค่าพารามิเตอร์ Canny, contour mode ฯลฯ)
- ลองปรับชุดเทคนิคหลายแบบ (binary threshold, Canny, morphological ops, retrieval mode ของ contour) เพื่อให้ได้ผลลัพธ์ที่ดีที่สุดสำหรับเอกสารเป้าหมาย

## ข้อควรระวัง / Edge cases
- เอกสารที่สแกนไม่สม่ำเสมอ (rotation, skew) อาจต้อง pre-processing เพิ่มเติม (deskew, rotate)
- เส้นตารางที่จางหรือขาดตอนอาจทำให้การเชื่อมต่อ contour ผิดพลาด — ปรับ kernel/morphology
- เซลล์ที่ว่างเปล่าหรือมี handwriting อาจต้อง pipeline แยกสำหรับ OCR แบบ Handwriting

## แนวทางการ debug สั้น ๆ
1. เปิดภาพระหว่างทาง (intermediate images) ในโฟลเดอร์ `output/` เพื่อตรวจสอบ mask/contour
2. เพิ่ม verbose/visualize ในโค้ด (cv2.imshow หรือเขียนไฟล์ภาพ) เพื่อดูผลแต่ละขั้นตอน
3. ใช้ชุดตัวอย่างจาก `dataset/` เพื่อทดลองค่าต่าง ๆ

## การมีส่วนร่วม (Contributing)
- เพิ่ม issue เพื่อแจ้งบั๊กหรือขอฟีเจอร์
- ถ้าส่ง PR: ระบุไฟล์ที่แก้ และแนบตัวอย่าง input/output ถ้าเป็นไปได้

## License
โปรเจคนี้ยังไม่มีการระบุ license ใน repository — หากต้องการเผยแพร่ให้เปิดไฟล์ `LICENSE` และเลือก license ที่ต้องการ (เช่น MIT)

## ติดต่อ / ผู้พัฒนา
- เจ้าของ repository: tanakrid

---
ถ้าต้องการ ผมสามารถปรับ README ให้มีตัวอย่างคำสั่งจริงจากสคริปต์ที่คุณใช้บ่อย ๆ หรือเพิ่มส่วน quick-start สำหรับ use-case เฉพาะ (เช่น ประมวลผล batch ของ PDF ทั้งโฟลเดอร์) ได้ — แจ้งสคริปต์ที่ต้องการให้ผมลงรายละเอียดได้เลย
