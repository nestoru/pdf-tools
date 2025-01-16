"""Tests for the OCR processor module."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pdf_tools.processors.ocr import OCRConfig, OCRResult, PDFOCRProcessor

@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir

@pytest.fixture
def sample_pdf(temp_dirs):
    """Create a sample PDF file in the input directory."""
    input_dir, _ = temp_dirs
    pdf_path = input_dir / "test.pdf"
    # Create an empty PDF file
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    return pdf_path

def test_ocr_config_validation():
    """Test OCR configuration validation."""
    # Test default configuration
    config = OCRConfig()
    assert config.languages == ['eng']
    assert config.dpi == 300
    assert config.clean is True

    # Test custom configuration
    config = OCRConfig(
        languages=['eng', 'fra'],
        dpi=400,
        clean=False
    )
    assert config.languages == ['eng', 'fra']
    assert config.dpi == 400
    assert config.clean is False

    # Test language code conversion
    config = OCRConfig(languages=['en', 'fr'])
    assert config.languages == ['eng', 'fra']

    # Test validation errors
    with pytest.raises(ValueError):
        OCRConfig(languages=[])  # Empty languages list

    with pytest.raises(ValueError):
        OCRConfig(dpi=50)  # DPI too low

def test_should_process(temp_dirs, sample_pdf):
    """Test should_process logic based on file timestamps."""
    input_dir, output_dir = temp_dirs
    processor = PDFOCRProcessor()

    # Test when output doesn't exist
    output_path = output_dir / "test.pdf"
    assert processor.should_process(sample_pdf, output_path) is True

    # Test when output is older than input
    output_path.write_bytes(b"%PDF-1.4\n%EOF")
    old_time = datetime.now() - timedelta(days=1)
    os.utime(output_path, (old_time.timestamp(), old_time.timestamp()))
    assert processor.should_process(sample_pdf, output_path) is True

    # Test when output is newer than input
    new_time = datetime.now() + timedelta(days=1)
    os.utime(output_path, (new_time.timestamp(), new_time.timestamp()))
    assert processor.should_process(sample_pdf, output_path) is False

def create_mock_pdf_reader():
    """Create a mock PDF reader with pages."""
    mock_reader = MagicMock()
    mock_pages = [MagicMock() for _ in range(3)]
    type(mock_reader).pages = mock_pages
    return mock_reader

def test_process_pdf(temp_dirs, sample_pdf):
    """Test processing a single PDF file."""
    input_dir, output_dir = temp_dirs
    output_path = output_dir / "test.pdf"

    mock_pdf_reader = create_mock_pdf_reader()
    
    # Using multiple context managers to mock all necessary components
    with patch('ocrmypdf.ocr') as mock_ocr, \
         patch('PyPDF2.PdfReader', return_value=mock_pdf_reader), \
         patch('builtins.open', mock_open()):
        
        # Mock successful OCR
        mock_ocr.return_value = 0
        
        processor = PDFOCRProcessor()
        result = processor.process_pdf(sample_pdf, output_path)
        
        # Verify OCR was called with correct parameters
        mock_ocr.assert_called_once()
        args, kwargs = mock_ocr.call_args
        assert kwargs['language'] == 'eng'
        assert result.success is True
        assert result.pages_processed == 3
        assert result.error_message is None

def test_process_directory(temp_dirs):
    """Test processing a directory of PDF files."""
    input_dir, output_dir = temp_dirs
    
    # Create multiple test PDFs
    for i in range(3):
        pdf_path = input_dir / f"test_{i}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    
    mock_pdf_reader = create_mock_pdf_reader()
    
    # Mock all file operations
    with patch('ocrmypdf.ocr') as mock_ocr, \
         patch('PyPDF2.PdfReader', return_value=mock_pdf_reader), \
         patch('builtins.open', mock_open()):
        
        # Mock successful OCR
        mock_ocr.return_value = 0
        
        processor = PDFOCRProcessor()
        results = processor.process_directory(input_dir, output_dir)
        
        assert results.processed_count == 3
        assert results.skipped_count == 0
        assert not results.errors
        assert mock_ocr.call_count == 3

def test_error_handling(temp_dirs):
    """Test error handling during processing."""
    input_dir, output_dir = temp_dirs
    pdf_path = input_dir / "test.pdf"
    pdf_path.write_bytes(b"not a valid pdf")
    
    with patch('ocrmypdf.ocr') as mock_ocr:
        # Mock OCR failure
        mock_ocr.side_effect = Exception("OCR failed")
        
        processor = PDFOCRProcessor()
        result = processor.process_pdf(pdf_path, output_dir / "test.pdf")
        
        assert result.success is False
        assert result.error_message is not None
        assert "OCR failed" in result.error_message

def test_multiple_languages():
    """Test OCR processing with multiple languages."""
    config = OCRConfig(languages=['eng', 'fra'])
    processor = PDFOCRProcessor(config=config)
    
    mock_pdf_reader = create_mock_pdf_reader()
    
    with patch('ocrmypdf.ocr') as mock_ocr, \
         patch('PyPDF2.PdfReader', return_value=mock_pdf_reader), \
         patch('builtins.open', mock_open()):
        
        mock_ocr.return_value = 0
        
        # Process a dummy PDF
        result = processor.process_pdf(
            Path('dummy.pdf'),
            Path('output.pdf')
        )
        
        # Verify correct language parameter
        args, kwargs = mock_ocr.call_args
        assert kwargs['language'] == 'eng+fra'
        assert result.success is True

def test_invalid_directory():
    """Test handling of invalid directory paths."""
    processor = PDFOCRProcessor()
    
    with pytest.raises(ValueError, match="Input directory does not exist"):
        processor.process_directory(
            Path("/nonexistent/path"),
            Path("/some/output")
        )
