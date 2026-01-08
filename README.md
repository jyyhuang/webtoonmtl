# Webtoon MTL (Machine Translation)

A Python tool for extracting and translating Korean text from webtoon images using OCR and neural machine translation.

## Features

- **OCR Extraction**: Extract Korean text from images using EasyOCR
- **Neural Translation**: Translate Korean to English using fine-tuned transformer models
- **Model Training**: Fine-tune translation models on Korean-English datasets
- **Batch Processing**: Process multiple images efficiently
- **Configurable Logging**: Comprehensive logging system with configurable output

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd webtoonmtl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Translation

```python
from mtlcore import MtlCore

# Initialize the core system
core = MtlCore()

# Process a single image
translations = core.process_image_to_translation("path/to/image.jpg")
print(translations)
```

### Training a Custom Model

```python
from translator import KoreanTranslator, TranslationConfig

# Create custom configuration
config = TranslationConfig(
    model_name="Helsinki-NLP/opus-mt-ko-en",
    num_train_epochs=5,
    learning_rate=3e-5
)

# Initialize and train
translator = KoreanTranslator(config)
translator.train()
```

### Using the Fine-tuned Model

```python
from translator import KoreanTranslator

# Load with fine-tuned model (automatically detected)
translator = KoreanTranslator()

# Translate text
korean_text = "안녕하세요"
english_translation = translator.translate(korean_text)
print(english_translation)  # "Hello"
```

## Project Structure

- `main.py` - Entry point with logging configuration
- `mtlcore.py` - Core OCR and translation pipeline
- `translator.py` - Neural translation model and training utilities
- `train_translator.py` - Training script for translation models
- `test.py` - Testing utilities
- `logging_config/` - Logging configuration files
- `tests/` - Unit tests

## Dependencies

Key dependencies include:
- `easyocr` - Korean OCR functionality
- `transformers` - Hugging Face transformer models
- `torch` - PyTorch deep learning framework
- `datasets` - Dataset loading and processing
- `evaluate` - Model evaluation metrics

## Configuration

The system uses a `TranslationConfig` dataclass for customization:

- `model_name`: Base Hugging Face model for translation
- `dataset_name`: Dataset for fine-tuning
- `max_length`: Maximum sequence length
- Training parameters (epochs, batch size, learning rate, etc.)
- Model saving and evaluation settings

## Logging

The project includes configurable logging with JSON-based configuration. Logs are saved to the `logs/` directory and can be customized via `logging_config/config.json`.

## License

See LICENSE file for details.
