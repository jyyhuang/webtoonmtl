import inspect
import logging
from dataclasses import asdict, dataclass

import numpy as np
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

bleu = None
chrf = None


@dataclass
class TrainingConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    dataset_name: str = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
    max_length: int = 128

    output_dir: str = "./models/fine-tuned-model"
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 2
    max_steps: int = 2000
    optim: str = "adafactor"
    metric_for_best_model: str = "eval_bleu"
    fp16: bool = True
    predict_with_generate: bool = True
    push_to_hub: bool = False


class ModelTrainer:
    def __init__(
        self,
        config: TrainingConfig | None = None,
    ) -> None:
        global bleu, chrf
        if bleu is None or chrf is None:
            import evaluate

            bleu = evaluate.load("sacrebleu")
            chrf = evaluate.load("chrf")

        self.config = config or TrainingConfig()
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

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def _prepare_datasets(self):
        dataset = load_dataset(self.config.dataset_name)
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

        dataset_validation = dataset["train"].shuffle().select(range(2000))
        dataset_test = dataset["test"].shuffle().select(range(2000))

        tokenized_dataset_validation = dataset_validation.map(
            self._preprocess_function,
            batched=True,
            batch_size=2,
            remove_columns=dataset_validation.column_names,
        )

        tokenized_dataset_test = dataset_test.map(
            self._preprocess_function,
            batched=True,
            batch_size=2,
            remove_columns=dataset_test.column_names,
        )

        logger.info("Tokenization of dataset completed")

        return tokenized_dataset_validation, tokenized_dataset_test

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
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.__tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.__tokenizer.pad_token_id)
        decoded_labels = self.__tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = chrf.compute(predictions=decoded_preds, references=decoded_labels)

        return {"bleu": bleu_result["score"], "chrf": chrf_result["score"]}

    def train(self) -> None:
        """
        Starts the fine-tuning process
        """
        logger.info("Model training starting...")

        tokenized_dataset_validation, tokenized_dataset_test = self._prepare_datasets()

        for parameter in self.__model.parameters():
            parameter.requires_grad = True
        num_layers_to_freeze = 10
        for layer_index, layer in enumerate(self.__model.model.encoder.layers):
            if (
                layer_index
                < len(self.__model.model.encoder.layers) - num_layers_to_freeze
            ):
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        num_layers_to_freeze = 10
        for layer_index, layer in enumerate(self.__model.model.decoder.layers):
            if (
                layer_index
                < len(self.__model.model.encoder.layers) - num_layers_to_freeze
            ):
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model.to(device)

        valid_args = inspect.signature(Seq2SeqTrainingArguments).parameters

        trainer_settings = {
            k: v for k, v in asdict(self.config).items() if k in valid_args
        }

        training_args = Seq2SeqTrainingArguments(**trainer_settings)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.__tokenizer, model=self.__model
        )

        trainer = Seq2SeqTrainer(
            model=self.__model,
            args=training_args,
            train_dataset=tokenized_dataset_test,
            eval_dataset=tokenized_dataset_validation,
            data_collator=data_collator,
            processing_class=self.__tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

        trainer.save_model(self.config.output_dir)
        self.__tokenizer.save_pretrained(self.config.output_dir)

        logger.info(f"Model saved to {self.config.output_dir}")
