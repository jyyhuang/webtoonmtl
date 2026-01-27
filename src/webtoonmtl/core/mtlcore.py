import logging
from pathlib import Path
from typing import cast
from webtoonmtl._utils.logger import setup_logging

from webtoonmtl.core.translator import KoreanTranslator

logger = logging.getLogger(__name__)

_logging_initialized = False


def init_logging_once():
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True


class MtlCore:

    def __init__(self):
        try:
            init_logging_once()
            self.__ocr_reader = None
            self.__translator = KoreanTranslator()

            logger.info("MtlCore initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MtlCore: {e}")
            raise

    def _ensure_ocr_loaded(self):
        """
        Ensures the EasyOCR reader is loaded.
        """

        import easyocr

        if self.__ocr_reader is None:
            logger.info("Loading EasyOCR model...")
            self.__ocr_reader = easyocr.Reader(["ko"])
            logger.info("EasyOCR model loaded.")

    def extract_with_ocr(self, file: str | Path) -> list[str]:
        """
        Extract Korean text from image using EasyOCR.

        Args:
            file: Path to the image file

        Returns:
            List of extracted Korean text strings
        """
        try:
            self._ensure_ocr_loaded()
            result = cast(
                list[str],
                self.__ocr_reader.readtext(str(file), detail=0, paragraph=True),
            )
            return result

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

    def image_to_translation(self, file_path: str | Path) -> str | list[str]:
        """
        Extract text from image and translate to English.

        Args:
            file_path: Path to the image file

        Returns:
            List of translated English strings
        """
        try:
            korean_text = self.extract_with_ocr(file_path)
            if not korean_text:
                logger.warning(f"No text extracted from {file_path}")
                return []
            logger.info(f"Extracted {len(korean_text)} text segments")
            translations = self.__translator.translate(korean_text)
            logger.info(f"Translated {len(translations)} segments")
            return translations
        except Exception as e:
            logger.error(f"Pipeline processing failed for {file_path}: {e}")
            raise
