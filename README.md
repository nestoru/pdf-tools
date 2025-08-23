# PDF Tools

A collection of PDF processing tools including OCR capabilities and translation with layout preservation.

## Features

- OCR processing for PDF files
- PDF translation with layout preservation
- Timestamp-based processing to avoid redundant operations
- Support for multiple languages
- Rich CLI interface with progress tracking
- Detailed logging and error reporting
- Regeneration of PDFs from existing translations

## Installation

### System Requirements

Before installing the package, ensure you have the following system dependencies:

- Python 3.11 or higher
- Tesseract OCR:
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils unpaper ghostscript`
  - macOS: `brew install tesseract tesseract-lang poppler unpaper ghostscript`

For additional languages, install the corresponding Tesseract language packs:
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr-[lang]` (e.g., `tesseract-ocr-fra` for French)
- macOS: Language packs are included with Tesseract

Poppler is also required for PDF processing:
- Ubuntu/Debian: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

### Installing Poetry

This project uses Poetry for dependency management. If you don't have Poetry installed, you can install it using one of these methods:

**Linux, macOS:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:
```bash
poetry --version
```

### Project Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-tools.git
cd pdf-tools
```

2. Install dependencies using Poetry:
```bash
poetry env use python3.11
poetry install
```

3. Activate the Poetry virtual environment:
```bash
poetry shell
```

You can later exit the virtual environment with:
```bash
exit
```

### Alternative Installation

If you prefer to install from PyPI (once published):
```bash
pip install pdf-tools
```

## Usage

### Command Line Interface

The tool provides two main commands: `ocr` and `translate`.

#### OCR Processing

Process PDF files with OCR:
```bash
# Basic usage
pdf-tools /input/dir /output/dir

# Advanced usage with language options (eng, fra for English and French)
pdf-tools ocr --lang eng --lang fra /input/dir /output/dir

# Force OCR even if text is detected
pdf-tools ocr --force-ocr /input/dir /output/dir

# Produce default pdf and in addition docx
pdf-tools ocr --formats pdf,docx --force-ocr /input/dir /output/dir

# Skip pages that already contain text
pdf-tools ocr --skip-text /input/dir /output/dir

# Adjust image quality
pdf-tools ocr --dpi 400 --clean /input/dir /output/dir
```

#### PDF Translation

Translate PDF files while preserving layout:
```bash
# Translate PDFs using OpenAI
pdf-tools translate --config config.json --ml-provider openai --ml-model gpt-4-turbo --target-lang spa /input/dir /output/dir

# Enable verbose logging
pdf-tools translate --config config.json --ml-provider openai --ml-model gpt-3.5-turbo --target-lang fra /input/dir /output/dir --verbose

# Regenerate PDFs from existing translations without re-translating
pdf-tools translate --config config.json --ml-provider openai --ml-model gpt-4-turbo --target-lang spa --regenerate-target /input/dir /output/dir
```

Both commands use timestamp-based processing to avoid redundant operations. Files will only be processed if they've been modified since the last translation or if no previous output exists.

### Translation Workflow

The translation process works as follows:

1. **Extraction**: Extracts text and layout information from PDFs into a structured JSON format
2. **Translation**: Translates the content using ML services while preserving layout information
3. **Generation**: Regenerates PDFs with translated text overlaid on the original document structure

When using the `--regenerate-target` option, only step 3 is performed. This is useful for:
- Regenerating PDFs after manually correcting translated JSON files
- Testing improvements to the PDF generation process without redoing translations
- Fixing PDF rendering issues without incurring additional translation costs

### Configuration File

Create a JSON configuration file for translation:

```json
{
  "ml_engine": {
    "api_key": "sk-your-api-key-here"
  },
  "translation": {
    "max_expansion_factor": 1.3,
    "max_contraction_factor": 0.7
  }
}
```

You can also set API keys as environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Python API

#### OCR Processing

```python
from pathlib import Path
from pdf_tools.processors.ocr import PDFOCRProcessor, OCRConfig

# Initialize processor with custom configuration
config = OCRConfig(
    languages=['en'],
    min_confidence=0.7,
    dpi=300
)
processor = PDFOCRProcessor(config=config)

# Process a directory
input_dir = Path('input')
output_dir = Path('output')
results = processor.process_directory(input_dir, output_dir)
print(f"Processed: {results.processed_count} files")
print(f"Skipped: {results.skipped_count} files")

# Process a single file
result = processor.process_pdf(
    input_path=Path('input/document.pdf'),
    output_path=Path('output/document.pdf')
)
if result.success:
    print(f"Processed {result.pages_processed} pages")
```

#### PDF Translation

```python
from pathlib import Path
from pdf_tools.processors.translator import PDFTranslator, TranslationConfig

# Initialize translator with configuration
config = TranslationConfig(
    ml_provider="openai",
    ml_model="gpt-4-turbo",
    target_language="es",
    api_key="your-api-key-here"
)
translator = PDFTranslator(config=config)

# Process a single PDF
result = translator.process_pdf(Path('input/document.pdf'))
print(f"Generated files: {result}")

# Process a directory
input_dir = Path('input')
output_dir = Path('output')
results = translator.process_directory(input_dir, output_dir)
print(f"Processed: {results.processed_count} files")
print(f"Skipped: {results.skipped_count} files")

# Generate PDF from existing translation (skip extraction and translation)
source_pdf = Path('input/document.pdf')
translated_json = Path('output/document_es.json')
target_pdf = Path('output/document_es.pdf')
translator.generate_pdf_from_json(translated_json, target_pdf)
```

## Development

### Common Poetry Commands

Here are some useful Poetry commands for development:

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show currently installed packages
poetry show

# Run tests through poetry
poetry run pytest

# Build the package
poetry build

# Publish to PyPI (when ready)
poetry publish
```

### Scheduling

Use just cron as follows:

```
* * * * * ( flock -n 10 || exit 0; source ~/.profile && cd ~/workspace/pdf-tools && poetry run pdf-tools ocr /mnt/mydrive/pdf-ocr/pdf-ocr-input /mnt/mydrive/pdf-ocr/pdf-ocr-output > /tmp/pdf-ocr.log ) 10>~/pdf-ocr.lock
```

### Running Tests

```bash
# Run tests using Poetry
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=pdf_tools

# Run tests in watch mode during development
poetry run pytest-watch
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Format code:
```bash
# Format code with black
poetry run black .

# Sort imports
poetry run isort .

# Run linting
poetry run flake8 .

# Run all formatting (if you create a Makefile or script)
poetry run make format
```

## Sharing the code
```
fdfind -H -t f --exclude '.git' --exclude 'poetry.lock' -0 | xargs -0 -I {} sh -c 'echo "File: {}"; cat {}'
```

## License

[MIT License](LICENSE)
