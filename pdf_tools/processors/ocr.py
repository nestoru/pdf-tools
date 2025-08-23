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
    output_formats: list[str] = Field(
        default_factory=lambda: ['pdf'],
        description="Output formats to generate (pdf, docx)"
    )

    @validator('languages')
    def validate_languages(cls, v):
        """Validate that at least one language is provided and they are ISO 639-2."""
        if not v:
            raise ValueError("At least one language must be specified")

        # Convert any 2-letter codes to 3-letter codes
        lang_map = {'en': 'eng', 'fr': 'fra', 'de': 'deu', 'es': 'spa'}
        return [lang_map.get(lang, lang) for lang in v]

    @validator('output_formats')
    def validate_output_formats(cls, v):
        """Validate output formats."""
        if not v:
            raise ValueError("At least one output format must be specified")
        
        valid_formats = {'pdf', 'docx'}
        invalid_formats = set(v) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid output formats: {', '.join(invalid_formats)}. Valid formats: {', '.join(valid_formats)}")
        
        return v

class OCRResult(BaseModel):
    """Result of OCR processing."""

    success: bool = Field(..., description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    pages_processed: int = Field(0, ge=0, description="Number of pages processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    pdf_generated: bool = Field(default=False, description="Whether PDF was generated")
    docx_generated: bool = Field(default=False, description="Whether DOCX was generated")

class DirectoryProcessResult(BaseModel):
    """Result of processing a directory of PDFs."""

    processed_count: int = Field(0, ge=0, description="Number of files processed")
    skipped_count: int = Field(0, ge=0, description="Number of files skipped")
    errors: list[str] = Field(default_factory=list, description="List of error messages")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    docx_generated: int = Field(0, ge=0, description="Number of DOCX files generated")

class PDFOCRProcessor:
    """Handles OCR processing for PDF files."""

    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize the OCR processor."""
        self.config = config or OCRConfig()
        logger.debug(f"Initialized with languages: {self.config.languages}")
        logger.debug(f"Output formats: {self.config.output_formats}")

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

    def convert_pdf_to_docx(self, pdf_path: Path, docx_path: Path) -> bool:
        """Convert PDF to DOCX with comprehensive formatting preservation."""
        try:
            from pdf2docx import Converter
            
            logger.info(f"Converting PDF to DOCX: {pdf_path} -> {docx_path}")
            
            # Ensure output directory exists
            docx_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert PDF to DOCX with advanced options for better formatting preservation
            cv = Converter(pdf_path)
            cv.convert(
                docx_path, 
                start=0, 
                end=None,
                # Advanced options for better conversion quality
                multi_processing=True,
                cpu_count=min(4, os.cpu_count() or 1),  # Use up to 4 cores but respect system limits
            )
            cv.close()
            
            # Verify the DOCX file was created and has content
            if not docx_path.exists():
                raise Exception("DOCX file was not created")
            
            if docx_path.stat().st_size == 0:
                raise Exception("DOCX file was created but is empty")
            
            logger.info(f"Successfully converted to DOCX: {docx_path}")
            return True
            
        except ImportError:
            logger.error("pdf2docx library not found. Install with: pip install pdf2docx")
            return False
        except Exception as e:
            logger.error(f"Error converting PDF to DOCX: {str(e)}")
            # Clean up empty DOCX file if it was created
            if docx_path.exists() and docx_path.stat().st_size == 0:
                try:
                    docx_path.unlink()
                    logger.info(f"Removed empty DOCX file: {docx_path}")
                except Exception:
                    pass
            return False

    def process_pdf(self, input_path: Path, output_dir: Path, relative_path: Path) -> OCRResult:
        """Process a single PDF file with OCR and save the result(s)."""
        start_time = datetime.now()
        pdf_generated = False
        docx_generated = False

        try:
            logger.info(f"Processing file: {input_path.absolute()}")

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate PDF output path
            pdf_output_path = output_dir / relative_path.parent / f"{relative_path.stem}.pdf"
            pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Process PDF with OCR if PDF format is requested
            if 'pdf' in self.config.output_formats:
                try:
                    result = ocrmypdf.ocr(
                        input_file=str(input_path),
                        output_file=str(pdf_output_path),
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

                        cmd.extend([str(input_path), str(pdf_output_path)])

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
                    if not pdf_output_path.exists():
                        raise Exception(f"OCR completed but PDF output file does not exist")

                    if pdf_output_path.stat().st_size == 0:
                        raise Exception(f"OCR completed but PDF output file is empty")

                    pdf_generated = True
                    logger.info(f"Successfully generated PDF: {pdf_output_path}")
                else:
                    error_msg = f"OCR failed with exit code: {result}"
                    logger.error(error_msg)
                    return OCRResult(
                        success=False,
                        error_message=error_msg,
                        pages_processed=0,
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        pdf_generated=False,
                        docx_generated=False
                    )

            # Convert to DOCX if requested
            if 'docx' in self.config.output_formats:
                # Use the OCR-ed PDF as source if it exists, otherwise use the original
                source_pdf = pdf_output_path if pdf_generated else input_path
                docx_output_path = output_dir / relative_path.parent / f"{relative_path.stem}.docx"
                
                if self.convert_pdf_to_docx(source_pdf, docx_output_path):
                    docx_generated = True

            # Get page count from processed or original PDF
            try:
                with open(pdf_output_path if pdf_generated else input_path, 'rb') as f:
                    from pypdf import PdfReader
                    pdf = PdfReader(f)
                    num_pages = len(pdf.pages)
            except Exception as e:
                logger.warning(f"Could not get page count: {str(e)}")
                num_pages = 0

            # Determine overall success
            requested_pdf = 'pdf' in self.config.output_formats
            requested_docx = 'docx' in self.config.output_formats
            
            success = True
            if requested_pdf and not pdf_generated:
                success = False
            if requested_docx and not docx_generated:
                success = False

            if success:
                formats_generated = []
                if pdf_generated:
                    formats_generated.append("PDF")
                if docx_generated:
                    formats_generated.append("DOCX")
                logger.info(f"Successfully processed file: {input_path.absolute()} -> {', '.join(formats_generated)}")
            
            return OCRResult(
                success=success,
                pages_processed=num_pages,
                processing_time=(datetime.now() - start_time).total_seconds(),
                pdf_generated=pdf_generated,
                docx_generated=docx_generated
            )

        except Exception as e:
            error_msg = f"Error processing {input_path.name}: {str(e)}"
            logger.error(error_msg)

            # If an error occurred, make sure to clean up any empty output files
            if 'pdf' in self.config.output_formats:
                pdf_output_path = output_dir / relative_path.parent / f"{relative_path.stem}.pdf"
                if pdf_output_path.exists() and pdf_output_path.stat().st_size == 0:
                    try:
                        pdf_output_path.unlink()
                        logger.info(f"Removed empty PDF file after error: {pdf_output_path.absolute()}")
                    except Exception:
                        pass
            
            if 'docx' in self.config.output_formats:
                docx_output_path = output_dir / relative_path.parent / f"{relative_path.stem}.docx"
                if docx_output_path.exists() and docx_output_path.stat().st_size == 0:
                    try:
                        docx_output_path.unlink()
                        logger.info(f"Removed empty DOCX file after error: {docx_output_path.absolute()}")
                    except Exception:
                        pass

            return OCRResult(
                success=False,
                error_message=error_msg,
                pages_processed=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                pdf_generated=False,
                docx_generated=False
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
        docx_generated_count = 0
        errors = []

        # Recursively process all PDF files
        for input_path in self.find_pdf_files(input_dir):
            # Skip input files with zero size
            if input_path.stat().st_size == 0:
                logger.warning(f"Skipping empty input file: {input_path.absolute()}")
                continue

            # Calculate the relative path
            relative_path = input_path.relative_to(input_dir)

            # Check if we should process this file
            pdf_output_path = output_dir / relative_path.parent / f"{relative_path.stem}.pdf"
            
            if self.should_process(input_path, pdf_output_path):
                result = self.process_pdf(input_path, output_dir, relative_path)
                if result.success:
                    processed_count += 1
                    if result.docx_generated:
                        docx_generated_count += 1
                else:
                    errors.append(result.error_message)
            else:
                logger.info(f"Skipping {input_path.name} - output is newer than input")
                skipped_count += 1

        return DirectoryProcessResult(
            processed_count=processed_count,
            skipped_count=skipped_count,
            errors=errors,
            total_processing_time=(datetime.now() - start_time).total_seconds(),
            docx_generated=docx_generated_count
        )
