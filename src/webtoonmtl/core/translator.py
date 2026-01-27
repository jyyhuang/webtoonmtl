import logging
from dataclasses import dataclass
from pathlib import Path
from webtoonmtl._utils.logger import setup_logging


logger = logging.getLogger(__name__)

_logging_initialized = False


def init_logging_once():
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True


@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    model_dir: str = "~/.webtoonmtl/model"


class KoreanTranslator:
    def __init__(
        self,
        config: TranslationConfig | None = None,
    ) -> None:

        init_logging_once()
        self.config = config or TranslationConfig()
        self.__tokenizer = None
        self.__model = None
        model_dir = Path(self.config.model_dir).expanduser().resolve()

        if (
            model_dir.exists()
            and (model_dir / "config.json").exists()
            and (model_dir / "tokenizer_config.json").exists()
            and (model_dir / "model.safetensors").exists()
        ):
            self.__load_path = str(model_dir)
        else:
            self.__load_path = self.config.model_name

        logger.info(f"Translator initialized to use path: {self.__load_path}")

    def _ensure_loaded(self) -> None:
        if self.__model is None and self.__tokenizer is None:
            print(f"Loading model: {self.__load_path}")
            self._load_model(self.__load_path)

    def _load_model(self, path: str | Path) -> None:
        """
        Internal loader for model

        Args:
            path: str or Path to a fine-tuned model, or HF base model

        Returns:
            None
        """
        try:

            from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

            self.__tokenizer = MarianTokenizer.from_pretrained(path)

            self.__model = AutoModelForSeq2SeqLM.from_pretrained(path)

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate(
        self,
        text: str | list[str],
    ) -> str | list[str]:
        """
        Translate Korean text to English.

        Args:
            text: Single or a list of Korean strings to translate

        Returns:
            String or list of translated English strings
        """
        if not text:
            return []

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._ensure_loaded()

        self.__model.eval()
        self.__model.to(device)

        with torch.no_grad():
            inputs = self.__tokenizer(text, return_tensors="pt", padding=True).to(
                device
            )

            outputs = self.__model.generate(**inputs, max_length=256)

            result = self.__tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return result[0] if isinstance(text, str) else result
