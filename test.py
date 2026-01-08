import pathlib
import time
import torch

from mtlcore import MtlCore
from translator import KoreanTranslator, TranslationConfig


def test_translation_workflow():
    """Test both single and batch translation using the public API."""
    print("=== Korean to English Translation Test ===")

    test_phrases = [
        "안녕하세요",  # Hello
        "오늘 날씨가 좋습니다",  # The weather is good today
        "만화",  # Comics/Manga
    ]

    print("Initializing KoreanTranslator...")
    start_time = time.time()

    # We can pass a custom config if we want to test with a small max_length
    translator = KoreanTranslator()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    print(
        f"Device being used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
    )

    print("Testing Batch Translation:")
    print("-" * 30)
    try:
        results = translator.translate(test_phrases)
        for ko, en in zip(test_phrases, results):
            print(f"KO: {ko} -> EN: {en}")
    except Exception as e:
        print(f"Batch translation failed: {e}")

    print("\nTesting Single String Translation:")
    print("-" * 30)
    # Main character
    single_result = translator.translate("주인공")
    print(f"KO: 주인공 -> EN: {single_result}")



def test_ocr():
    """Test OCR."""
    print("=== OCR Test ===\n")
    image_path = pathlib.Path("images/test01.jpg")
    if not image_path.exists():
        print(f"Skipping OCR test: {image_path} not found.")
        return
    mtl = MtlCore()
    extraction = mtl.process_image_to_translation(image_path)
    print(extraction)


if __name__ == "__main__":
    print("Korean Manga Translation System Test\n")

    # 1. Test standard translation
    test_translation_workflow()

    print("\n=== Test Complete ===")
