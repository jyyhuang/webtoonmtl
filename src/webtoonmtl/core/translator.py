import logging
import torch
from dataclasses import dataclass
from pathlib import Path
from accelerate import Accelerator

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    model_dir: str = "~/.webtoonmtl/model"


class KoreanTranslator:
    def __init__(
        self,
        config: TranslationConfig | None = None,
    ) -> None:

        self.config = config or TranslationConfig()
        self.__tokenizer = None
        self.__model = None
        self.__pipeline = None
        self.__device = None

        self.__device = Accelerator().device

        model_dir = Path(self.config.model_dir).expanduser().resolve()

        has_weights = (model_dir / "model.safetensors").exists() or (
            model_dir / "config.json"
        ).exists()

        if model_dir.exists() and has_weights:
            load_path = str(model_dir)
        else:
            load_path = self.config.model_name

        print(f"Loading model: {load_path}")
        self._load_model(load_path)

    def _load_model(self, path: str | Path) -> None:
        """
        Internal loader for model

        Args:
            path: str or Path to a fine-tuned model, or HF base model

        Returns:
            None
        """
        try:
            is_local = Path(path).exists()

            self.__tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=is_local
            )

            self.__model = AutoModelForSeq2SeqLM.from_pretrained(
                path, local_files_only=is_local
            )

            self.__model.to(self.__device)

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def _get_pipeline(self):
        """Get or create pipeline for specified language"""
        if self.__pipeline is None:
            self.__pipeline = pipeline(
                "translation",
                model=self.__model,
                tokenizer=self.__tokenizer,
                device=self.__device,
            )

        return self.__pipeline

    def translate(
        self,
        text: str | list[str],
    ) -> list[str]:
        """
        Translate Korean text to English.

        Args:
            text: Single or a list of Korean strings to translate

        Returns:
            List of translated English strings
        """
        if not text:
            return []

        if self.__tokenizer is None or self.__model is None:
            raise RuntimeError("Model not loaded")

        texts = [text] if isinstance(text, str) else text

        self.__model.eval()
        with torch.no_grad():
            translator = self._get_pipeline()
            outputs = translator(texts, max_length=512, truncation=True)

        return [o["translation_text"] for o in outputs]
