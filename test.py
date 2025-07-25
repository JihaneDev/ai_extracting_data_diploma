import os
import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image

# Config
POPPLER_PATH = r"C:\Users\hp\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin"
FOLDER_DIPLOME = "DIPLOME/"

# Charger Donut
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Fonction traitement d'un diplôme
def traiter_diplome(pdf_path):
    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    image = images[0].convert("RGB")

    pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values

    task_prompt = (
    "<s_docvqa><s_question>Quelle est la filière ?<s_answer>"
)


    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=512
    )

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Tenter de parser le JSON
    try:
        data = json.loads(result)
    except Exception:
        # Si Donut ne produit pas un JSON propre
        data = {"raw_output": result}

    return data

# Parcourir le dossier DIPLOME
resultats = {}

for file in os.listdir(FOLDER_DIPLOME):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(FOLDER_DIPLOME, file)
        print(f"Traitement de : {file}")
        infos = traiter_diplome(pdf_path)
        resultats[file] = infos

# Sauvegarder en JSON
with open("resultats_diplomes.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, indent=4, ensure_ascii=False)

print("Extraction terminée. Résultats enregistrés dans resultats_diplomes.json.")