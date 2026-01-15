# Webtoon MTL

A Python tool for extracting and translating Korean text from webtoon images using OCR and neural machine translation.

## Project Status
Still under development, expect breaking changes

## Table of Contents
- [Webtoon MTL](#webtoon-mtl)
    - [Table of Contents](#table-of-contents)
    - [Demo](#demo)
    - [Background](#background)
    - [Features](#features)
    - [Installation](#installation)
    - [Usage](#usage)
        - [Basic Translation](#basic-translation)
        - [Training a Custom Model](#training-a-custom-model)
        - [Using the Fine-tuned Model](#using-the-fine-tuned-model)
    - [Roadmap](#roadmap)
    - [License](#license)
  
## Demo

## Background
After years of reading webtoons and manhwa, I repeatedly ran into the same frustration of reaching the latest available chapter in English, only to find that newer chapters exist only in Korean. While fan translations eventually appear, they are often delayed or incomplete.

With recent advances in optical character recognition (OCR) and neural machine translation (NMT), I decided it was finally time to address this problem myself. This project is personal, but feel free to use it too!

## Features

- **OCR Extraction**: Extract Korean text from images using EasyOCR
- **Neural Translation**: Translate Korean to English using transformer models
- **Model Training**: Fine-tune translation models on Korean-English datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jyyhuang/webtoonmtl.git
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
# Use mtlcore like in previous example (automatically detected)
# Or if you just want the translator
translator = KoreanTranslator()

# Translate text
korean_text = "안녕하세요"
english_translation = translator.translate(korean_text)
```

## Roadmap
- ✅ Extract text from images using OCR
- ✅ Use transformer to translate Korean text
- ✅ Add support for fine-tuning the translation model
- ⬜ Create better command line usage
- ⬜ Add better testing
- ⬜ Cache OCR, model, and translations
- ⬜ Improve logging, error handling, and progress reports
- ⬜ Desktop GUI with PyQT
- ⬜ Package as a pip library

## License

See LICENSE file for details.
