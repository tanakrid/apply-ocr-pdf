from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from pdf2image import convert_from_path

def resize_if_needed(img, max_size):
    width, height = img.size
    # Only resize if one dimension exceeds max_size
    if width > 300 or height > 300:
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, int(height * scale))
        else:
            scale = max_size / float(height)
            new_size = (int(width * scale), max_size)

        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"{width, height}==> {img.size}")
        return img
    else:
        return img 


model = AutoModelForImageTextToText.from_pretrained(
    "scb10x/typhoon-ocr1.5-2b", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("scb10x/typhoon-ocr1.5-2b")

images = convert_from_path("test.pdf")  # <-- ตรงนี้แทน image.png

# เลือกหน้าแรก (จะวนหลายหน้าก็ได้)
img = images[0]

#This is important because the model is trained with a fixed image dimension of 1800 px
img = resize_if_needed(img, 1800)

prompt = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax.
- Images: Wrap visual elements in <figure> tags.
"""

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ],
        }
    ]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=10000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
