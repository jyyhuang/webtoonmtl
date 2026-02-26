import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SETUP_DIR = Path.home() / ".webtoonmtl"
SETUP_COMPLETE_FILE = SETUP_DIR / ".setup_complete"
MODEL_DIR = SETUP_DIR / "model"


def is_setup_complete() -> bool:
    return SETUP_COMPLETE_FILE.exists()


def mark_setup_complete() -> None:
    SETUP_DIR.mkdir(parents=True, exist_ok=True)
    SETUP_COMPLETE_FILE.touch()
    logger.info("Setup marked as complete")


def get_translation_model_status() -> tuple[bool, str]:
    model_dir = MODEL_DIR.expanduser().resolve()
    required_files = ["config.json", "tokenizer_config.json", "model.safetensors"]

    if not model_dir.exists():
        return False, "Model directory does not exist"

    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        return False, f"Missing files: {', '.join(missing)}"

    return True, "Translation model ready"


def download_ocr_model(
    progress_callback: Callable[[int, str], None] | None = None,
) -> bool:
    try:
        import easyocr

        if progress_callback:
            progress_callback(0, "Initializing EasyOCR download...")

        logger.info("Downloading EasyOCR Korean model...")

        reader = easyocr.Reader(["ko"], download_enabled=True, verbose=False)

        if progress_callback:
            progress_callback(100, "EasyOCR model ready")

        del reader
        return True

    except Exception as e:
        logger.error(f"Failed to download OCR model: {e}")
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        return False


def download_translation_model(
    progress_callback: Callable[[int, str], None] | None = None,
) -> bool:
    try:
        from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-ko-en"
        model_dir = MODEL_DIR.expanduser().resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(0, "Downloading tokenizer...")

        logger.info("Downloading tokenizer...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_dir))

        if progress_callback:
            progress_callback(50, "Downloading model...")

        logger.info("Downloading translation model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(str(model_dir))

        if progress_callback:
            progress_callback(100, "Translation model ready")

        del model
        del tokenizer
        return True

    except Exception as e:
        logger.error(f"Failed to download translation model: {e}")
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        return False


def run_full_setup(
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> bool:
    success = True

    if progress_callback:
        progress_callback("ocr", 0, 0)

    if not download_ocr_model(
        lambda p, m: progress_callback("ocr", p, 0) if progress_callback else None
    ):
        success = False

    if success:
        if progress_callback:
            progress_callback("translation", 0, 0)

        if not download_translation_model(
            lambda p, m: (
                progress_callback("translation", p, 0) if progress_callback else None
            )
        ):
            success = False

    if success:
        mark_setup_complete()

    return success
