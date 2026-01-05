import pathlib
import time

from mtlcore import MtlCore
from translator import KoreanTranslator


def test_basic_translation():
    """Test basic Korean to English translation."""
    print("=== Korean to English Translation Test ===\n")

    test_phrases = [
        "안녕하세요",  # Hello
        "오늘 날씨가 좋습니다",  # The weather is good today
        "만화",  # Comics/Manga
        "주인공",  # Main character
        "사랑합니다",  # I love you
    ]

    print("Testing KoreanTranslator class...")
    start_time = time.time()
    translator = KoreanTranslator()
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds\n")

    print("\nTest Phrases and Translations:")
    print("-" * 40)

    try:
        result = translator.translate_ko_to_en(test_phrases)
        print(result)
    except Exception as e:
        print(f"Batch translation failed: {e}")

def test_ocr():
    """Test OCR."""
    print("=== OCR Test ===\n")
    image_path = pathlib.Path("images/test01.jpg")
    mtl = MtlCore()
    extraction = mtl.process_image_to_translation(image_path)
    print(extraction)


if __name__ == "__main__":
    print("Korean Manga Translation System Test\n")
    print("This test will verify the translation functionality.\n")

    # Test basic translation
    test_basic_translation()

    print("\n=== Test Complete ===")
    print("If you see translations above, the system is working!")
    print(
        "For image-based translation, provide image paths to process_image_to_translation()"
    )
