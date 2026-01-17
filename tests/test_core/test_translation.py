"""
Tests for Korean to English translation functionality.
"""

import logging
import unittest

from translator import KoreanTranslator


class TestKoreanTranslator(unittest.TestCase):
    """Test cases for KoreanTranslator class."""

    @classmethod
    def setUpClass(cls):
        """
        Load the model once for all tests.
        This avoids reloading the model for every test.
        """
        cls.translator = KoreanTranslator()

    def setUp(self):
        self.test_cases = [
            "안녕하세요",
            "만화",
            "주인공",
            "학교",
            "사랑",
        ]

    def test_single_translation(self):
        """Translating a single Korean string returns one non-empty English string."""
        result = self.translator.translate("안녕하세요")

        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())

    def test_batch_translation(self):
        """Batch translation preserves order and length."""
        results = self.translator.translate(self.test_cases)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.test_cases))

        for output in results:
            self.assertIsInstance(output, str)
            self.assertTrue(output.strip())

    def test_empty_input(self):
        """Empty input returns an empty list."""
        result = self.translator.translate([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
