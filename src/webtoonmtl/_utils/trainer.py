#!pip install evaluate
#!pip install torch
#!pip install datasets
#!pip install transformers
#!pip install sacrebleu
#!pip install sacremoses

import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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

@dataclass
class TrainingConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    dataset_name: str = "lemon-mint/Korean-FineTome-100k"
    max_length: int = 128

    output_dir: Path = Path("~/.webtoonmtl/model").expanduser().resolve()
    output_data_dir: Path = Path("~/.webtoonmtl/data").expanduser().resolve()
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    num_train_epochs: int = 5
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


def load_model(model_name: str, gradient_checkpointing: bool = True):
    """
    Load tokenizer and model from HuggingFace.

    Args:
        model_name: str path to HF base model
        gradient_checkpointing: whether to enable gradient checkpointing

    Returns:
        tuple: (tokenizer, model)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        return tokenizer, model
    except Exception as e:
        raise


def tokenize_function(examples, tokenizer, max_length: int = 128):
    """
    Tokenize examples for translation.

    Args:
        examples: batch of examples from dataset
        tokenizer: tokenizer to use
        max_length: max sequence length

    Returns:
        dict: tokenized inputs
    """
    inputs, targets = [], []

    for msgs in examples["messages"]:
        if "content" in msgs[0] and "content_en" in msgs[0]:
            inputs.append(msgs[0]["content"])
            targets.append(msgs[0]["content_en"])
        else:
            inputs.append("")
            targets.append("")

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    return model_inputs


def prepare_datasets(dataset_name: str, tokenizer, max_length: int = 128):
    """
    Load and tokenize dataset.

    Args:
        dataset_name: name of dataset to load
        tokenizer: tokenizer to use
        max_length: max sequence length

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=0.2)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenized_dataset_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_dataset_test = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    return tokenized_dataset_train, tokenized_dataset_test


def compute_metrics(eval_preds, tokenizer, bleu_metric, chrf_metric):
    """
    Compute BLEU and chrF metrics for evaluation.

    Args:
        eval_preds: tuple of (predictions, labels)
        tokenizer: tokenizer for decoding
        bleu_metric: BLEU metric object
        chrf_metric: chrF metric object

    Returns:
        dict: metrics dictionary
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]

    bleu_result = bleu_metric.compute(
        predictions=decoded_preds, references=decoded_labels
    )
    chrf_result = chrf_metric.compute(
        predictions=decoded_preds, references=decoded_labels
    )

    return {"bleu": bleu_result["score"], "chrf": chrf_result["score"]}


def train(
    config: TrainingConfig | None = None,
    resume_from_checkpoint: str | None = None,
):
    """
    Starts the fine-tuning process.

    Args:
        config: Training configuration
        resume_from_checkpoint: checkpoint path to resume from
    """
    config = config or TrainingConfig()


    tokenizer, model = load_model(config.model_name, config.gradient_checkpointing)

    train_dataset, test_dataset = prepare_datasets(
        config.dataset_name, tokenizer, config.max_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    valid_args = inspect.signature(Seq2SeqTrainingArguments).parameters

    trainer_settings = {k: v for k, v in asdict(config).items() if k in valid_args}

    training_args = Seq2SeqTrainingArguments(**trainer_settings)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, bleu_metric, chrf_metric
        ),
        processing_class=tokenizer,
    )

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        output_dir = config.output_dir
        output_data_dir = config.output_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_data_dir.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(output_dir))

        metrics = train_result.metrics
        eval_metrics = trainer.evaluate()
        metrics.update({k: v for k, v in eval_metrics.items()})

        metrics_path = output_data_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)


    except Exception as e:
        raise
