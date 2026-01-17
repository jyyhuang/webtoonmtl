import logging
import torch
from dataclasses import dataclass
from pathlib import Path
from accelerate import Accelerator

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    model_name: str = "facebook/nllb-200-distilled-600M"
    model_dir: str = "./models/fine-tuned-model"


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

            self.__model = AutoModelForSeq2SeqLM.from_pretrained(path)

            self.__model.to(self.__device)
            self.__model.eval()

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def _get_pipeline(self, src_lang: str, tgt_lang: str):
        """Get or create pipeline for specified language"""
        if self.__pipeline is None:
            self.__pipeline = pipeline(
                "translation",
                model=self.__model,
                tokenizer=self.__tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                device=self.__device,
            )

        return self.__pipeline

    def translate(
        self,
        text: str | list[str],
        src_lang: str = "kor_Hang",
        tgt_lang: str = "eng_Latn",
    ) -> list[str]:
        """
        Translate Korean text to English.

        Args:
            text: Single or a list of Korean strings to translate
            src_lang: Source language code (defualt is Korean)
            tgt_lang: Target language code (defualt is English)

        Returns:
            List of translated English strings
        """
        if not text:
            return []

        if self.__tokenizer is None or self.__model is None:
            raise RuntimeError("Model not loaded")

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        with torch.no_grad():
            translator = self._get_pipeline(src_lang, tgt_lang)
            outputs = translator(texts, max_length=400)

        translations = [o["translation_text"] for o in outputs]

        return translations
