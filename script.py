from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torch
import logging
import sys
from transformers.trainer_utils import get_last_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize model and processor
        logger.info("Initializing model and processor...")
        model_name = "naver-clova-ix/donut-base"
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Configure model settings
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_doc>"])[0]
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.use_cache = False  # Important for training

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)

        task_prompt = "<s_diplome>"

        class DonutDataset(Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return {
                    "pixel_values": self.images[idx],
                    "labels": self.labels[idx]
                }

        def load_examples(folder):
            images = []
            labels = []
            
            for filename in sorted(os.listdir(folder)):
                if filename.lower().endswith((".jpeg", ".jpg", ".png")):
                    try:
                        image_path = os.path.join(folder, filename)
                        image = Image.open(image_path).convert("RGB")
                        # Resize to more manageable dimensions
                        image = image.resize((1280, 960), Image.Resampling.LANCZOS)
                        images.append(image)

                        json_name = os.path.splitext(filename)[0] + ".json"
                        json_path = os.path.join(folder, json_name)
                        
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                json_obj = json.load(f)
                            labels.append(task_prompt + json.dumps(json_obj["gt_parse"]))
                        else:
                            labels.append(task_prompt)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        continue
            
            return images, labels

        logger.info("Loading datasets...")
        train_images, train_labels = load_examples("diplome_dataset/train")
        val_images, val_labels = load_examples("diplome_dataset/val")

        train_dataset = DonutDataset(train_images, train_labels)
        val_dataset = DonutDataset(val_images, val_labels)

        output_dir = os.path.abspath("./donut_diplome_model")
        os.makedirs(output_dir, exist_ok=True)

        def collate_fn(batch):
            images = [item["pixel_values"] for item in batch]
            labels = [item["labels"] for item in batch]

            pixel_values = processor(images, return_tensors="pt").pixel_values
            
            input_ids = processor.tokenizer(
                labels,
                add_special_tokens=False,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            
            input_ids[input_ids == processor.tokenizer.pad_token_id] = -100

            return {
                "pixel_values": pixel_values,
                "labels": input_ids
            }

        # Check for existing checkpoint
        last_checkpoint = None
        if os.path.isdir(output_dir):
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is not None:
                logger.info(f"Found checkpoint: {last_checkpoint}")

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            predict_with_generate=True,
            logging_steps=1,
            save_steps=5,
            num_train_epochs=5,
            save_total_limit=2,
            logging_dir="./logs",
            report_to="none",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=5,
            disable_tqdm=False,
            use_cpu=True,  # Explicitly use CPU
            optim="adamw_torch",  # Specify optimizer
            dataloader_num_workers=0,  # Set to 0 for stability on Windows
        )

        logger.info("Initializing trainer...")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
        )

        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info("Training completed successfully!")

        # Save final model
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info(f"Model and training artifacts saved to {output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()