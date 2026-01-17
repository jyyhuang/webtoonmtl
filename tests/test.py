import pathlib
import time

from ..src.webtoonmtl import KoreanTranslator, MtlCore


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

    translator = KoreanTranslator()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    print("Testing Batch Translation:")
    print("-" * 30)
    try:
        results = translator.translate(test_phrases)
        for ko, en in zip(test_phrases, results):
            print(f"KO: {ko} -> EN: {en}")
    except Exception as e:
        print(f"Batch translation failed: {e}")

def test_ocr():
    """Test OCR."""
    print("=== OCR Test ===\n")
    image_path = pathlib.Path("images/test01.jpg")
    if not image_path.exists():
        print(f"Skipping OCR test: {image_path} not found.")
        return
    mtl = MtlCore()
    extraction = mtl.image_to_translation(image_path)
    print(extraction)


if __name__ == "__main__":
    print("Korean Manga Translation System Test\n")

    # 1. Test standard translation
    test_translation_workflow()
    test_ocr()

    print("\n=== Test Complete ===")
