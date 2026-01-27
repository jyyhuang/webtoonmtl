import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from webtoonmtl._utils.logger import setup_logging

import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)

_logging_initialized = False


def init_logging_once():
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True


@dataclass
class TrainingConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    dataset_name: str = "lemon-mint/Korean-FineTome-100k"
    max_length: int = 128

    output_dir: Path = Path("~/.webtoonmtl/model").expanduser().resolve()
    output_data_dir: Path = Path("~/.webtoonmtl/data").expanduser().resolve()
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    num_train_epochs: int = 3
    optim: str = "adamw_torch"
    metric_for_best_model: str = "bleu"
    fp16: bool = True
    predict_with_generate: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    push_to_hub: bool = False


class ModelTrainer:
    def __init__(
        self,
        config: TrainingConfig | None = None,
    ) -> None:

        init_logging_once()
        self.config = config or TrainingConfig()

        self.bleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")

        self.__tokenizer = None
        self.__model = None

        self._load_model(self.config.model_name)

    def _load_model(self, path: str) -> None:
        """
        Internal loader for model

        Args:
            path: str to HF base model
        """
        try:
            self.__tokenizer = AutoTokenizer.from_pretrained(path)
            self.__model = AutoModelForSeq2SeqLM.from_pretrained(path)

            if self.config.gradient_checkpointing:
                self.__model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def _prepare_datasets(self):
        dataset = load_dataset(self.config.dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.2)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        tokenized_dataset_train = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        tokenized_dataset_test = test_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names,
        )

        logger.info(
            f"Tokenization completed - Train: {len(tokenized_dataset_train)}, Test: {len(tokenized_dataset_test)}"
        )

        return tokenized_dataset_train, tokenized_dataset_test

    def _tokenize_function(self, examples):
        inputs, targets = [], []

        for msgs in examples["messages"]:
            if "content" in msgs[0] and "content_en" in msgs[0]:
                inputs.append(msgs[0]["content"])
                targets.append(msgs[0]["content_en"])
            else:
                inputs.append("")
                targets.append("")

        model_inputs = self.__tokenizer(
            inputs,
            text_target=targets,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
        )

        return model_inputs

    def _compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.__tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.__tokenizer.pad_token_id)
        decoded_labels = self.__tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu_result = self.bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        chrf_result = self.chrf.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        return {"bleu": bleu_result["score"], "chrf": chrf_result["score"]}

    def train(self, resume_from_checkpoint: str | None = None) -> None:
        """
        Starts the fine-tuning process
        """
        logger.info("Model training starting...")

        tokenized_dataset_train, tokenized_dataset_test = self._prepare_datasets()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.__model.to(device)

        valid_args = inspect.signature(Seq2SeqTrainingArguments).parameters

        # training args setup in config
        trainer_settings = {
            k: v for k, v in asdict(self.config).items() if k in valid_args
        }

        training_args = Seq2SeqTrainingArguments(**trainer_settings)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.__tokenizer, model=self.__model
        )

        # trainer
        trainer = Seq2SeqTrainer(
            model=self.__model,
            args=training_args,
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_test,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            processing_class=self.__tokenizer,
        )

        try:
            # train loop
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # save model to output_dir
            output_dir = self.config.output_dir
            output_data_dir = self.config.output_data_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_data_dir.mkdir(parents=True, exist_ok=True)

            trainer.save_model(str(output_dir))
            logger.info(f"Model saved to {output_dir}")

            # save metrics
            metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            metrics.update({k: v for k, v in eval_metrics.items()})

            metrics_path = output_data_dir / "metrics.json"
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Evaluation results: {eval_metrics}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
