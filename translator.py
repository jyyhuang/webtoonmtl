import logging
import evaluate
import torch
import numpy as np
import json
from datasets import load_dataset
from dataclasses import dataclass, asdict
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

logger = logging.getLogger(__name__)
metric = evaluate.load("sacrebleu")


@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    dataset_name: str = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
    max_length: int = 128

    output_dir: str = "./models/fine_tuned_translator"
    gradient_checkpointing: bool = True
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    logging_steps: int = 100
    optim: str = "adafactor"
    metric_for_best_model: str = "bleu"
    fp16: bool = True
    predict_with_generate: bool = True
    push_to_hub: bool = False
    load_best_model_at_end: bool = True
    greater_is_better: bool = True


class KoreanTranslator:
    def __init__(
        self,
        config: TranslationConfig | None = None,
    ) -> None:
        self.config = config or TranslationConfig()

        self.__tokenizer = None
        self.__model = None
        self.__pipeline = None
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if Path(self.config.output_dir).exists():
            self._load_model(self.config.output_dir)
        else:
            self._load_model(self.config.model_name)

    def _load_model(self, path: str | Path) -> None:
        """
        Internal loader for model

        Args:
            path: str or Path to a fine-tuned model, or HF base model

        Returns:
            None
        """
        try:
            self.__tokenizer = AutoTokenizer.from_pretrained(path)

            self.__model = AutoModelForSeq2SeqLM.from_pretrained(path)
            self.__model.to(self.__device)

            if self.config.gradient_checkpointing:
                self.__model.gradient_checkpointing_enable()
                self.__model.config.use_cache = False

            self.__pipeline = None

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def _prepare_datasets(self):
        # Split and tokenize dataset
        dataset = load_dataset(self.config.dataset_name)
        dataset = dataset["train"].train_test_split(
            test_size=0.2, seed=42
        )  # features: english, korean, score

        tokenized_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        logger.info("Tokenization of dataset completed")

        return tokenized_dataset

    def _preprocess_function(self, examples):
        inputs = examples["korean"]
        targets = examples["english"]

        model_inputs = self.__tokenizer(
            inputs,
            text_target=targets,
            max_length=self.config.max_length,
            truncation=True,
        )

        return model_inputs

    def _compute_metrics(self, eval_preds):
        # Compute bleu metric
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.__tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.__tokenizer.pad_token_id)
        decoded_labels = self.__tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Basic stripping for cleanup
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    def _save_current_model(self):
        self.__model.save_pretrained(self.config.output_dir)
        self.__tokenizer.save_pretrained(self.config.output_dir)
        # Save config
        with open(f"{self.config.output_dir}/training_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info("Current model saved")

    def train(self, resume_from_checkpoint: bool | None = None) -> None:
        """
        Starts the fine-tuning process

        Args:
            resume_from_checkpoint: bool that indicates if training should resume from a checkpoint
        """

        logger.info("Model training starting...")
        tokenized_datasets = self._prepare_datasets()

        # Ensure model is in training mode
        self.__model.train()
        if self.config.gradient_checkpointing:
            self.__model.config.use_cache = False

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            gradient_checkpointing=self.config.gradient_checkpointing,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            metric_for_best_model=self.config.metric_for_best_model,
            fp16=self.config.fp16,
            predict_with_generate=self.config.predict_with_generate,
            push_to_hub=self.config.push_to_hub,
            load_best_model_at_end=self.config.load_best_model_at_end,
            greater_is_better=self.config.greater_is_better,
            generation_max_length=self.config.max_length,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.__tokenizer, model=self.__model
        )

        trainer = Seq2SeqTrainer(
            model=self.__model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            processing_class=self.__tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.__pipeline = None
        self._save_current_model()
        logger.info("Model training finished")

    def translate(self, text: str | list[str]) -> str | list[str]:
        """
        Translate Korean text to English.

        Args:
            text: Single or a list of Korean strings to translate

        Returns:
            Single or a list of translated English strings
        """
        if not text:
            return []

        input_is_list = isinstance(text, list)
        texts = text if input_is_list else [text]

        self.__model.config.use_cache = True
        self.__model.eval()

        try:
            if self.__pipeline is None:
                self.__pipeline = pipeline(
                    "translation_ko_to_en",
                    model=self.__model,
                    tokenizer=self.__tokenizer,
                    device=self.__device,
                )

            with torch.inference_mode():
                results = self.__pipeline(texts)
                translations = [t["translation_text"] for t in results]
            return translations if input_is_list else translations[0]

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
