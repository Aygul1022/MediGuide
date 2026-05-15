from datasets import load_dataset
import json
import os

print("Orijinal veri seti Hugging Face'ten indiriliyor. Bu işlem internet hızınıza göre 1-2 dakika sürebilir...")

# HuggingFace'ten gerçek veri setini indiriyoruz
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
train_data = dataset['train']

# data klasörünün olduğundan emin olalım
os.makedirs("data", exist_ok=True)

# Verileri bizim ajanımızın okuyacağı JSONL formatına çevirip kaydediyoruz
output_path = "data/clinical_qa_processed.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in train_data:
        # Orijinal verideki alanları bizim formata uyarlıyoruz
        record = {
            "question": item.get("instruction", "") + " " + item.get("input", ""),
            "answer": item.get("output", ""),
            "category": "Genel Tıp", # Orijinal veride kategori ayrımı yok, varsayılan atıyoruz
            "source": "Medical Meadow Flashcards"
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Harika! {len(train_data)} adet gerçek klinik vaka kaydı başarıyla indirildi ve '{output_path}' konumuna kaydedildi!")