import os
import json
import logging
import sys
import torch
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
import cv2
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("donut_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration Poppler pour Windows
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + POPPLER_PATH


class DonutPDFDataset(Dataset):
    """Dataset spécialisé pour les PDF avec vérification complète"""
    
    def __init__(self, pdf_dir, label_dir, processor, max_samples=None, dpi=300):
        self.processor = processor
        self.dpi = dpi
        self.samples = []
        
        # Vérification initiale de poppler
        self._verify_poppler()
        self._prepare_dataset(pdf_dir, label_dir, max_samples)
        
        if len(self.samples) == 0:
            raise ValueError("Aucun échantillon valide trouvé")

    def _verify_poppler(self):
        """Vérification approfondie de poppler"""
        try:
            test_pdf = os.path.join("dataset", "train", "pdfs", "1_DIPLOME.pdf")
            images = convert_from_path(
                test_pdf,
                dpi=100,
                poppler_path=POPPLER_PATH,
                first_page=1,
                last_page=1
            )
            if not images:
                raise RuntimeError("Test de conversion échoué")
        except Exception as e:
            logger.error(f"ERREUR POPPLER: {str(e)}")
            raise RuntimeError(
                "Configuration Poppler invalide. Vérifiez que:\n"
                f"1. {POPPLER_PATH} contient pdftoppm.exe\n"
                "2. Les PDF ne sont pas protégés/corrompus\n"
                "3. Vous avez redémarré le terminal après modification du PATH"
            )

    def _prepare_dataset(self, pdf_dir, label_dir, max_samples):
        """Charge les données avec validation rigoureuse"""
        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')])
        
        for i, pdf_file in enumerate(pdf_files[:max_samples] if max_samples else pdf_files):
            base_name = os.path.splitext(pdf_file)[0]
            pdf_path = os.path.join(pdf_dir, pdf_file)
            label_path = os.path.join(label_dir, f"{base_name}.json")
            
            if not os.path.exists(label_path):
                logger.warning(f"Label manquant: {pdf_file}")
                continue
                
            try:
                # Validation du PDF
                if os.path.getsize(pdf_path) == 0:
                    raise ValueError("PDF vide")
                    
                # Validation du JSON
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    if "gt_parse" not in label_data:
                        raise ValueError("Champ gt_parse manquant")
                        
                self.samples.append({
                    "pdf_path": pdf_path,
                    "label_path": label_path
                })
            except Exception as e:
                logger.error(f"Erreur {pdf_file}: {str(e)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Conversion PDF->image à la volée"""
        sample = self.samples[idx]
        try:
            # Conversion PDF
            images = convert_from_path(
                sample["pdf_path"],
                dpi=self.dpi,
                poppler_path=POPPLER_PATH
            )
            if not images:
                raise ValueError("Conversion échouée")
            image = images[0].convert("RGB")
            
            # Traitement Donut
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            
            with open(sample["label_path"], 'r', encoding='utf-8') as f:
                labels = json.load(f)["gt_parse"]
                
            text = json.dumps(labels)
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            return {
                "pixel_values": pixel_values,
                "labels": labels
            }
            
        except Exception as e:
            logger.error(f"Erreur échantillon {idx}: {str(e)}")
            # Retourne un tensor vide
            return {
                "pixel_values": torch.zeros((3, 1920, 2560)),
                "labels": torch.zeros((512,), dtype=torch.long)
            }

def train_model():
    """Fonction principale d'entraînement"""
    try:
        logger.info("=== INITIALISATION ===")
        
        # 1. Configuration matérielle
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Dispositif: {device}")
        
        # 2. Chargement modèle
        from transformers import VisionEncoderDecoderModel, DonutProcessor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

        # Set the decoder_start_token_id required for training
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.max_length = 512
        model.to(device)
        
        print("decoder_start_token_id:", model.config.decoder_start_token_id)

        # 3. Chargement données
        logger.info("Chargement datasets...")
        train_dataset = DonutPDFDataset(
            pdf_dir="dataset/train/pdfs",
            label_dir="dataset/train/labels",
            processor=processor,
            dpi=300
        )
        
        val_dataset = DonutPDFDataset(
            pdf_dir="dataset/val/pdfs",
            label_dir="dataset/val/labels",
            processor=processor,
            max_samples=min(20, len(train_dataset)//5),
            dpi=300
        )
        
        logger.info(f"Échantillons: {len(train_dataset)} train, {len(val_dataset)} val")

        # 4. Configuration entraînement
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2 if device=="cuda" else 1,
            per_device_eval_batch_size=1,
            num_train_epochs=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            learning_rate=4e-5,
            warmup_steps=100,
            remove_unused_columns=False,
            fp16=(device=="cuda"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            save_total_limit=2,
            dataloader_num_workers=0  # Obligatoire pour Windows
        )

        # 5. Entraînement
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        logger.info("=== DÉBUT ENTRAÎNEMENT ===")
        trainer.train()
        logger.info("=== ENTRAÎNEMENT TERMINÉ ===")
        
        # 6. Sauvegarde
        model.save_pretrained("./results/final_model")
        processor.save_pretrained("./results/final_model")
        logger.info("Modèle sauvegardé")

    except Exception as e:
        logger.error(f"ERREUR: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Vérification préalable
    if not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm.exe")):
        logger.error(f"Poppler introuvable dans {POPPLER_PATH}")
        sys.exit(1)
        
    train_model()