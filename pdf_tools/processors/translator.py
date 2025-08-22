# pdf_tools/processors/translator.py
"""PDF translation with direct content stream manipulation."""

import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%m/%d/%y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TranslationConfig(BaseModel):
    """Configuration for PDF translation."""

    ml_provider: str = Field(
        default="openai",
        description="ML provider (openai, anthropic)"
    )

    ml_model: str = Field(
        default="gpt-4-turbo",
        description="ML model to use"
    )

    target_language: str = Field(
        default="",
        description="Target language code (ISO 639-2)"
    )

    api_key: str = Field(
        default="",
        description="API key for ML provider"
    )
    
    keep_postscript: bool = Field(
        default=False,
        description="Whether to keep PostScript files for debugging"
    )

    @validator('ml_provider')
    def validate_provider(cls, v):
        """Validate the ML provider."""
        valid_providers = ['openai', 'anthropic']
        if v not in valid_providers:
            raise ValueError(f"ML provider must be one of: {', '.join(valid_providers)}")
        return v

    @validator('target_language')
    def validate_language(cls, v):
        """Validate the target language code."""
        if not v:
            raise ValueError("Target language must be specified")
        return v

    @classmethod
    def from_file(cls, config_path: Path) -> 'TranslationConfig':
        """Load configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Extract and validate API key
        ml_engine = config_data.get('ml_engine', {})
        api_key = ml_engine.get('api_key', '')

        if not api_key:
            # Check environment variables for API key
            api_key = os.environ.get('OPENAI_API_KEY', '')
            if not api_key:
                api_key = os.environ.get('ANTHROPIC_API_KEY', '')

        if not api_key:
            raise ValueError("API key must be provided in config file or environment variables")

        # Extract translation config
        translation_config = config_data.get('translation', {})
        
        # Get keep_postscript setting if available
        keep_postscript = config_data.get('keep_postscript', False)

        return cls(
            api_key=api_key,
            keep_postscript=keep_postscript
        )


class TranslationResult(BaseModel):
    """Result of translation processing."""
    success: bool = Field(..., description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    processing_time: float = Field(..., description="Processing time in seconds")


class DirectoryProcessResult(BaseModel):
    """Result of processing a directory of PDFs."""
    processed_count: int = Field(0, ge=0, description="Number of files processed")
    skipped_count: int = Field(0, ge=0, description="Number of files skipped")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    total_processing_time: float = Field(..., description="Total processing time in seconds")


class MLTranslationProvider:
    """Base class for ML translation providers."""

    def __init__(self, config: TranslationConfig):
        """Initialize the ML translation provider."""
        self.config = config

    def translate_content_stream(self, content_stream: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a PDF content stream.
        
        Args:
            content_stream: PDF content stream to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated content stream
        """
        raise NotImplementedError("Subclasses must implement this method")


class PDFTranslator:
    """Handles translation processing for PDF files."""

    def __init__(self, config: TranslationConfig):
        """Initialize the PDF translator."""
        self.config = config
        self.provider = self._create_provider()

    def _create_provider(self) -> MLTranslationProvider:
        """Create the ML provider based on configuration."""
        if self.config.ml_provider == 'openai':
            from pdf_tools.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(self.config)
        elif self.config.ml_provider == 'anthropic':
            # For future implementation
            raise ValueError("Anthropic provider is not yet implemented")
        else:
            raise ValueError(f"Unsupported ML provider: {self.config.ml_provider}")

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

    def extract_content_streams(self, pdf_path: Path) -> Dict[int, str]:
        """
        Extract content streams from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to content streams
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for content extraction. Install with: pip install pymupdf")
            
        content_streams = {}
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Extracting content streams from {pdf_path} ({len(doc)} pages)")
            
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                # Extract raw content stream
                content = page.read_contents()
                # Decode to string if it's binary
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                content_streams[page_idx] = content
                
                # Optionally save as PS file
                if self.config.keep_postscript:
                    ps_file = pdf_path.parent / f"{pdf_path.stem}_page{page_idx+1}.ps"
                    with open(ps_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Saved PS file: {ps_file}")
                    
            doc.close()
            logger.info(f"Successfully extracted {len(content_streams)} content streams")
            return content_streams
        except Exception as e:
            logger.error(f"Error extracting content streams: {str(e)}")
            traceback.print_exc()
            return {}
    
    def create_translated_pdf(self, 
                             input_path: Path, 
                             output_path: Path, 
                             translated_streams: Dict[int, str]) -> bool:
        """
        Create a translated PDF file from content streams.
        
        Args:
            input_path: Path to the original PDF
            output_path: Path to save the translated PDF
            translated_streams: Dictionary mapping page numbers to translated content streams
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF creation. Install with: pip install pymupdf")
            
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                temp_pdf_path = temp_dir_path / "temp.pdf"

                # Copy original PDF to work with
                import shutil
                shutil.copy2(input_path, temp_pdf_path)

                # Open the PDF
                doc = fitz.open(str(temp_pdf_path))
                
                # Update each page's content
                for page_idx, content in translated_streams.items():
                    if page_idx >= len(doc):
                        logger.warning(f"Page index {page_idx} out of range for document with {len(doc)} pages")
                        continue
                        
                    page = doc[page_idx]
                    
                    # Convert string to bytes if needed
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    
                    # Replace the page's content stream
                    page.set_contents(content)
                
                # Save to temporary location
                temp_output_path = temp_dir_path / "output.pdf"
                doc.save(
                    str(temp_output_path),
                    garbage=4,        # Maximum garbage collection
                    deflate=True,     # Compress streams
                    clean=True,       # Clean content streams
                )
                doc.close()
                
                # Create output directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy to final destination using binary to avoid issues
                with open(temp_output_path, 'rb') as src:
                    with open(output_path, 'wb') as dst:
                        dst.write(src.read())
                
                logger.info(f"Successfully created translated PDF: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating translated PDF: {str(e)}")
            traceback.print_exc()
            return False
    
    def find_ps_files(self, pdf_path: Path) -> List[Path]:
        """
        Find PostScript files related to a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PostScript file paths
        """
        ps_files = list(pdf_path.parent.glob(f"{pdf_path.stem}_page*_translated.ps"))
        return ps_files
    
    def regenerate_from_ps_files(self, ps_files: List[Path], output_path: Path) -> bool:
        """
        Regenerate a PDF from PostScript files.
        
        Args:
            ps_files: List of PostScript file paths
            output_path: Path to save the regenerated PDF
            
        Returns:
            True if successful, False otherwise
        """
        # Requires PyMuPDF
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF regeneration. Install with: pip install pymupdf")
            
        try:
            # Create a new PDF document
            doc = fitz.open()
            
            # Sort PS files by page number
            ps_files.sort(key=lambda p: int(p.stem.split('_page')[1].split('_')[0]))
            
            # Process each PS file
            for ps_idx, ps_file in enumerate(ps_files):
                try:
                    # Extract page number from filename
                    page_pattern = ps_file.stem.split('_page')[1].split('_')[0]
                    page_idx = int(page_pattern) - 1  # Convert to 0-based index
                    
                    # Read content stream
                    with open(ps_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create a new page
                    page = doc.new_page()
                    
                    # Set content stream
                    page.set_contents(content.encode('utf-8'))
                    
                except Exception as e:
                    logger.error(f"Error processing PS file {ps_file}: {str(e)}")
                    continue
            
            # Save the PDF
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"Successfully regenerated PDF from PostScript files: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error regenerating PDF from PostScript files: {str(e)}")
            traceback.print_exc()
            return False

    def process_pdf_file(self, 
                        input_path: Path, 
                        output_path: Path, 
                        source_lang: str = "en", 
                        regenerate: bool = False) -> bool:
        """
        Process a single PDF file.
        
        Args:
            input_path: Path to the input PDF
            output_path: Path to save the output PDF
            source_lang: Source language code
            regenerate: Whether to regenerate from existing PS files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if regenerating from existing PS files
            if regenerate:
                ps_files = self.find_ps_files(output_path)
                if ps_files:
                    logger.info(f"Found {len(ps_files)} PS files for regeneration")
                    # Try to regenerate from PS files
                    if self.regenerate_from_ps_files(ps_files, output_path):
                        logger.info(f"Successfully regenerated PDF from existing PS files: {output_path}")
                        return True
                    else:
                        logger.warning("Failed to regenerate from PS files, falling back to fresh translation")
            
            # Extract content streams from input PDF
            content_streams = self.extract_content_streams(input_path)
            if not content_streams:
                raise ValueError(f"Failed to extract content streams from {input_path}")
            
            # Translate each content stream
            translated_streams = {}
            for page_idx, content in content_streams.items():
                logger.info(f"Translating content stream for page {page_idx + 1}")
                try:
                    translated_content = self.provider.translate_content_stream(
                        content, source_lang, self.config.target_language)
                    translated_streams[page_idx] = translated_content
                    
                    # Optionally save translated PS file
                    if self.config.keep_postscript:
                        ps_file = output_path.parent / f"{output_path.stem}_page{page_idx+1}_translated.ps"
                        with open(ps_file, 'w', encoding='utf-8') as f:
                            f.write(translated_content)
                        logger.debug(f"Saved translated PS file: {ps_file}")
                except Exception as e:
                    logger.error(f"Error translating page {page_idx + 1}: {str(e)}")
                    # Use original content as fallback
                    translated_streams[page_idx] = content
            
            # Create translated PDF
            return self.create_translated_pdf(input_path, output_path, translated_streams)
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            traceback.print_exc()
            return False

    def find_pdf_files(self, directory: Path) -> List[Path]:
        """Find all PDF files in a directory, case-insensitive for extension."""
        pdf_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(file_path)
        return sorted(pdf_files)

    def process_directory(self, input_dir: Path, output_dir: Path, regenerate_target: bool = False) -> DirectoryProcessResult:
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

            # Calculate the relative path and construct the output paths
            relative_path = input_path.relative_to(input_dir)
            base_name = relative_path.stem
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            target_pdf_path = output_subdir / f"{base_name}_{self.config.target_language}.pdf"

            # Check if processing is needed based on timestamps
            if regenerate_target or self.should_process(input_path, target_pdf_path):
                try:
                    # Process the PDF file
                    logger.info(f"Processing PDF file: {input_path}")
                    if self.process_pdf_file(input_path, target_pdf_path, "en", regenerate_target):
                        processed_count += 1
                    else:
                        error_msg = f"Failed to process {input_path.name}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error processing {input_path.name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            else:
                logger.info(f"Skipping {input_path.name} - output is newer than input")
                skipped_count += 1

        return DirectoryProcessResult(
            processed_count=processed_count,
            skipped_count=skipped_count,
            errors=errors,
            total_processing_time=(datetime.now() - start_time).total_seconds()
        )
