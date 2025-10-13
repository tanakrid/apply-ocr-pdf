from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("./trocr-thai-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-thai-finetuned")

image = Image.open("sample_handwritten.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
