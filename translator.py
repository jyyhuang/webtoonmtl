import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    max_length: int = 512
    num_beams: int = 4


class KoreanTranslator:

    def __init__(self, config: TranslationConfig | None = None) -> None:
        self.config = config or TranslationConfig()

        try:
            self.__tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            self.__model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name, dtype="auto", device_map="auto"
            )

            self.__model.eval()

            logger.info("Translation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate_ko_to_en(self, strings: list[str]) -> list[str]:
        """
        Translate Korean text to English.

        Args:
            strings: List of Korean strings to translate

        Returns:
            List of translated English strings
        """
        if not strings:
            return []

        try:
            model_inputs = self.__tokenizer(
                strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.__model.device)

            with torch.inference_mode():
                translated_tokens = self.__model.generate(
                    **model_inputs,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    early_stopping=True,
                )

            return self.__tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
