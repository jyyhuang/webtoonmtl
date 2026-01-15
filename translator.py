import logging
import torch
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    model_dir: str = "./models/fine-tuned-model"


class KoreanTranslator:
    def __init__(
        self,
        config: TranslationConfig | None = None,
    ) -> None:

        self.config = config or TranslationConfig()
        self.__tokenizer = None
        self.__model = None

        if self._is_valid_model_dir(Path(self.config.model_dir)):
            load_path = self.config.model_dir
        else:
            load_path = self.config.model_name
        print(f"Loading model: {load_path}")
        self._load_model(load_path)

    def _is_valid_model_dir(self, path: Path) -> bool:
        return path.exists() and (path / "config.json").exists()

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

            self.__model = AutoModelForSeq2SeqLM.from_pretrained(
                path, dtype="auto", device_map="auto"
            )

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate(self, text: str | list[str]) -> list[str]:
        """
        Translate Korean text to English.

        Args:
            text: Single or a list of Korean strings to translate

        Returns:
            Single or a list of translated English strings
        """
        if not text:
            return []

        if self.__tokenizer is None or self.__model is None:
            raise RuntimeError("Model not loaded")

        model_inputs = self.__tokenizer(
            text, padding=True, return_tensors="pt"
        ).to(self.__model.device)

        self.__model.eval()
        with torch.no_grad():
            translated_tokens = self.__model.generate(
                **model_inputs, max_length=256
            )

        outputs = self.__tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )

        return outputs
