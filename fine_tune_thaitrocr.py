from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
from PIL import Image

model_name = "suchut/thaitrocr-base-handwritten-beta1"

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# โหลด dataset จาก CSV
dataset = load_dataset("csv", data_files={"train": "thai_ocr_dataset/labels.csv"})

def preprocess(batch):
    images = [Image.open(f"thai_ocr_dataset/train/{path}").convert("RGB") for path in batch["filename"]]
    labels = processor.tokenizer(batch["text"], padding="max_length", truncation=True).input_ids
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

train_dataset = dataset["train"].map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-thai-finetuned",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    save_steps=500,
    logging_steps=100,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
model.save_pretrained("./trocr-thai-finetuned")
processor.save_pretrained("./trocr-thai-finetuned")
