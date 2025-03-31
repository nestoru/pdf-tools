# pdf_tools/processors/ocr.py

import logging
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import ocrmypdf
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%m/%d/%y %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

class OCRConfig(BaseModel):
    """Configuration for OCR processing."""

    languages: list[str] = Field(
        default_factory=lambda: ['eng'],
        description="Languages to use for OCR (ISO 639-2 codes)"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion"
    )
    force_ocr: bool = Field(
        default=False,
        description="Force OCR on every page, even if text is detected"
    )
    skip_text: bool = Field(
        default=False,
        description="Skip OCR on pages that already contain text"
    )
    rotate_pages: bool = Field(
        default=True,
        description="Automatically rotate pages based on detected text orientation"
    )
    clean: bool = Field(
        default=True,
        description="Clean visual noise from scanned images"
    )

    @validator('languages')
    def validate_languages(cls, v):
        """Validate that at least one language is provided and they are ISO 639-2."""
        if not v:
            raise ValueError("At least one language must be specified")

        # Convert any 2-letter codes to 3-letter codes
        lang_map = {'en': 'eng', 'fr': 'fra', 'de': 'deu', 'es': 'spa'}
        return [lang_map.get(lang, lang) for lang in v]

class OCRResult(BaseModel):
    """Result of OCR processing."""

    success: bool = Field(..., description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    pages_processed: int = Field(0, ge=0, description="Number of pages processed")
    processing_time: float = Field(..., description="Processing time in seconds")

class DirectoryProcessResult(BaseModel):
    """Result of processing a directory of PDFs."""

    processed_count: int = Field(0, ge=0, description="Number of files processed")
    skipped_count: int = Field(0, ge=0, description="Number of files skipped")
    errors: list[str] = Field(default_factory=list, description="List of error messages")
    total_processing_time: float = Field(..., description="Total processing time in seconds")

class PDFOCRProcessor:
    """Handles OCR processing for PDF files."""

    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize the OCR processor."""
        self.config = config or OCRConfig()
        logger.debug(f"Initialized with languages: {self.config.languages}")

    def should_process(self, input_path: Path, output_path: Path) -> bool:
        """Check if the PDF needs to be processed based on timestamps."""
        if not output_path.exists():
            return True

        # Check if output file is empty
        if output_path.stat().st_size == 0:
            logger.warning(f"Output file exists but is empty: {output_path.absolute()}")
            return True

        input_time = datetime.fromtimestamp(input_path.stat().st_mtime)
        output_time = datetime.fromtimestamp(output_path.stat().st_mtime)

        return input_time > output_time

    def process_pdf(self, input_path: Path, output_path: Path) -> OCRResult:
        """Process a single PDF file with OCR and save the result."""
        start_time = datetime.now()

        try:
            logger.info(f"Processing file: {input_path.absolute()}")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Process PDF with OCR using standard approach
            try:
                result = ocrmypdf.ocr(
                    input_file=str(input_path),
                    output_file=str(output_path),
                    language='+'.join(self.config.languages),
                    deskew=True,
                    clean=self.config.clean,
                    rotate_pages=self.config.rotate_pages,
                    force_ocr=self.config.force_ocr,
                    skip_text=self.config.skip_text,
                    progress_bar=False,
                    optimize=1,
                    output_type='pdf',
                    use_threads=True,
                    jobs=os.cpu_count()
                )
            except Exception as e:
                # If standard approach fails with image size error, try command line approach
                if "Image size" in str(e) and "pixels exceeds limit" in str(e):
                    logger.warning(f"Image size limit exceeded, trying command line approach")

                    cmd = [
                        "ocrmypdf",
                        "--language", "+".join(self.config.languages),
                        "--deskew",
                        "--optimize", "1",
                    ]

                    if self.config.clean:
                        cmd.append("--clean")
                    if self.config.rotate_pages:
                        cmd.append("--rotate-pages")
                    if self.config.force_ocr:
                        cmd.append("--force-ocr")
                    if self.config.skip_text:
                        cmd.append("--skip-text")

                    cmd.extend([str(input_path), str(output_path)])

                    logger.info(f"Running command: {' '.join(cmd)}")

                    try:
                        process = subprocess.run(
                            cmd,
                            env={
                                "OCRMYPDF_PIKEPDF_IMAGE_MAX_PIXELS": "2000000000",  # 2 billion pixels
                                "OCRMYPDF_IMAGE_MAX_PIXELS": "2000000000",  # Ensure this is set for any possible overrides
                            },
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        result = 0  # Success
                    except subprocess.CalledProcessError as e:
                        raise Exception(f"Command line OCR failed: {str(e)}")
                else:
                    # Re-raise other errors
                    raise

            if result == 0:  # Success
                # Verify the output file exists and has content
                if not output_path.exists():
                    raise Exception(f"OCR completed but output file does not exist")

                if output_path.stat().st_size == 0:
                    raise Exception(f"OCR completed but output file is empty")

                # Get page count from output PDF
                with open(output_path, 'rb') as f:
                    from pypdf import PdfReader
                    pdf = PdfReader(f)
                    num_pages = len(pdf.pages)

                logger.info(f"Successfully processed file: {input_path.absolute()} with {num_pages} pages")
                return OCRResult(
                    success=True,
                    pages_processed=num_pages,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            else:
                error_msg = f"OCR failed with exit code: {result}"
                logger.error(error_msg)
                return OCRResult(
                    success=False,
                    error_message=error_msg,
                    pages_processed=0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

        except Exception as e:
            error_msg = f"Error processing {input_path.name}: {str(e)}"
            logger.error(error_msg)

            # If an error occurred, make sure to clean up any empty output file
            if output_path.exists() and output_path.stat().st_size == 0:
                try:
                    output_path.unlink()
                    logger.info(f"Removed empty output file after error: {output_path.absolute()}")
                except Exception:
                    pass

            return OCRResult(
                success=False,
                error_message=error_msg,
                pages_processed=0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def find_pdf_files(self, directory: Path) -> List[Path]:
        """Find all PDF files in a directory, case-insensitive for extension."""
        pdf_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(file_path)
        return pdf_files

    def process_directory(self, input_dir: Path, output_dir: Path) -> DirectoryProcessResult:
        """Process all PDF files in a directory, including subdirectories."""
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()
        processed_count = 0
        skipped_count = 0
        errors = []

        # Recursively process all PDF files
        for input_path in self.find_pdf_files(input_dir):
            # Skip input files with zero size
            if input_path.stat().st_size == 0:
                logger.warning(f"Skipping empty input file: {input_path.absolute()}")
                continue

            # Calculate the relative path and construct the output path
            relative_path = input_path.relative_to(input_dir)

            # Ensure output filename has lowercase .pdf extension
            output_filename = relative_path.stem + '.pdf'
            output_path = output_dir / relative_path.parent / output_filename

            # Ensure the output directory structure exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self.should_process(input_path, output_path):
                result = self.process_pdf(input_path, output_path)
                if result.success:
                    processed_count += 1
                else:
                    errors.append(result.error_message)
            else:
                logger.info(f"Skipping {input_path.name} - output is newer than input")
                skipped_count += 1

        return DirectoryProcessResult(
            processed_count=processed_count,
            skipped_count=skipped_count,
            errors=errors,
            total_processing_time=(datetime.now() - start_time).total_seconds()
        )

