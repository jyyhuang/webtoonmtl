# Webtoon MTL

A Python application for extracting and translating Korean text from webtoon images using OCR and neural machine translation.

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
  - [Roadmap](#roadmap)
  - [License](#license)

## Demo
https://github.com/user-attachments/assets/6e52bf87-991e-4488-935f-eeeb72c1f76a  

## Background

After years of reading webtoons and manhwa, I repeatedly ran into the same frustration of reaching the latest available chapter in English, only to find that newer chapters exist only in Korean. While fan translations eventually appear, they are often delayed or incomplete.

With recent advances in optical character recognition (OCR) and neural machine translation (NMT), I decided it was finally time to address this problem myself. This project is personal, but feel free to use it too!

## Features

- **Simple GUI**: Desktop gui with PyQT6
- **OCR Extraction**: Extract Korean text from images using EasyOCR
- **Neural Translation**: Translate Korean to English using transformer models
- **Model Training**: Fine-tuned translation models on Korean-English datasets

## Installation

```bash
pip install webtoonmtl
```

## Usage

```
webtoonmtl start
```

## Roadmap

- ✅ Extract text from images using OCR
- ✅ Use transformer to translate Korean text
- ✅ Add support for fine-tuning the translation model
- ✅ Create command line usage
- ✅ Desktop GUI with PyQT
- ✅ Package as a pip library
- ⬜ Add better testing
- ⬜ Cache OCR, model, and translations
- ⬜ Improve logging, error handling, and progress reports

## License

See LICENSE file for details.
