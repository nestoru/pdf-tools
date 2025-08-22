# tests/test_translation.py
"""Tests for the PDF translation module."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pdf_tools.processors.translator import (
    TranslationConfig, TranslationResult, PDFTranslator, OpenAIProvider
)


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


@pytest.fixture
def sample_json(temp_dirs):
    """Create a sample JSON structure in the input directory."""
    input_dir, _ = temp_dirs
    json_path = input_dir / "test.json"
    
    # Create a simple JSON structure
    structure = {
        "metadata": {
            "originalFileName": "test.pdf",
            "language": "en",
            "creationDate": "2023-01-01"
        },
        "pages": [
            {
                "pageNumber": 1,
                "width": 612,
                "height": 792,
                "elements": [
                    {
                        "id": "text-001",
                        "type": "text",
                        "content": "Hello World",
                        "bbox": [100, 100, 300, 120],
                        "style": {
                            "fontFamily": "Arial",
                            "fontSize": 12,
                            "fontWeight": "normal",
                            "fontStyle": "normal",
                            "color": "#000000",
                            "alignment": "left"
                        },
                        "baselineY": 110
                    }
                ]
            }
        ]
    }
    
    # Write the JSON structure to file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(structure, json_file, indent=2)
    
    return json_path


@pytest.fixture
def config_file(temp_dirs):
    """Create a sample configuration file."""
    input_dir, _ = temp_dirs
    config_path = input_dir / "config.json"
    
    # Create a sample configuration
    config = {
        "ml_engine": {
            "api_key": "test-api-key"
        },
        "translation": {
            "max_expansion_factor": 1.3,
            "max_contraction_factor": 0.7
        }
    }
    
    # Write the configuration to file
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(config, config_file, indent=2)
    
    return config_path


def test_translation_config_validation():
    """Test translation configuration validation."""
    # Test default configuration
    config = TranslationConfig(api_key="test-key", target_language="es")
    assert config.ml_provider == "openai"
    assert config.ml_model == "gpt-4-turbo"
    assert config.target_language == "es"
    
    # Test custom configuration
    config = TranslationConfig(
        ml_provider="openai",
        ml_model="gpt-4",
        target_language="fra",
        api_key="test-key",
        max_expansion_factor=1.5
    )
    assert config.ml_provider == "openai"
    assert config.ml_model == "gpt-4"
    assert config.target_language == "fra"
    assert config.max_expansion_factor == 1.5
    
    # Test validation errors
    with pytest.raises(ValueError):
        TranslationConfig(ml_provider="invalid", api_key="test-key", target_language="es")
    
    with pytest.raises(ValueError):
        TranslationConfig(api_key="test-key", target_language="")


def test_config_from_file(config_file):
    """Test loading configuration from file."""
    # Test with existing file
    config = TranslationConfig.from_file(config_file)
    assert config.api_key == "test-api-key"
    assert config.max_expansion_factor == 1.3
    assert config.max_contraction_factor == 0.7
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        TranslationConfig.from_file(Path("nonexistent.json"))


def test_should_process(temp_dirs, sample_pdf):
    """Test should_process logic based on file timestamps."""
    input_dir, output_dir = temp_dirs
    config = TranslationConfig(api_key="test-key", target_language="es")
    
    with patch('pdf_tools.processors.translator.OpenAIProvider'):
        translator = PDFTranslator(config)
        
        # Test when output doesn't exist
        output_path = output_dir / "test_es.pdf"
        assert translator.should_process(sample_pdf, output_path) is True
        
        # Test when output is older than input
        output_path.write_bytes(b"%PDF-1.4\n%EOF")
        old_time = datetime.now() - timedelta(days=1)
        os.utime(output_path, (old_time.timestamp(), old_time.timestamp()))
        assert translator.should_process(sample_pdf, output_path) is True
        
        # Test when output is newer than input
        new_time = datetime.now() + timedelta(days=1)
        os.utime(output_path, (new_time.timestamp(), new_time.timestamp()))
        assert translator.should_process(sample_pdf, output_path) is False


def test_create_provider():
    """Test provider creation based on configuration."""
    # Test OpenAI provider
    config = TranslationConfig(ml_provider="openai", api_key="test-key", target_language="es")
    with patch('pdf_tools.processors.translator.OpenAIProvider') as mock_openai:
        translator = PDFTranslator(config)
        assert translator.provider is not None
        mock_openai.assert_called_once()
    
    # Test Anthropic provider (not implemented)
    config = TranslationConfig(ml_provider="anthropic", api_key="test-key", target_language="es")
    with pytest.raises(NotImplementedError):
        translator = PDFTranslator(config)


def test_openai_provider():
    """Test OpenAI provider translation."""
    config = TranslationConfig(
        ml_provider="openai",
        ml_model="gpt-4-turbo",
        target_language="es",
        api_key="test-key"
    )
    
    # Mock the requests module
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "pageNumber": 1,
                        "width": 612,
                        "height": 792,
                        "elements": [
                            {
                                "id": "text-001",
                                "type": "text",
                                "content": "Hola Mundo",
                                "bbox": [100, 100, 300, 120],
                                "style": {
                                    "fontFamily": "Arial",
                                    "fontSize": 12,
                                    "fontWeight": "normal",
                                    "fontStyle": "normal",
                                    "color": "#000000",
                                    "alignment": "left"
                                },
                                "baselineY": 110
                            }
                        ]
                    })
                }
            }
        ]
    }
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        provider = OpenAIProvider(config)
        
        # Test translation
        source_json = {
            "metadata": {
                "language": "en"
            },
            "pages": [
                {
                    "pageNumber": 1,
                    "width": 612,
                    "height": 792,
                    "elements": [
                        {
                            "id": "text-001",
                            "type": "text",
                            "content": "Hello World",
                            "bbox": [100, 100, 300, 120],
                            "style": {
                                "fontFamily": "Arial",
                                "fontSize": 12,
                                "fontWeight": "normal",
                                "fontStyle": "normal",
                                "color": "#000000",
                                "alignment": "left"
                            },
                            "baselineY": 110
                        }
                    ]
                }
            ]
        }
        
        translated_json = provider.translate(source_json, "es")
        
        # Verify API was called properly
        mock_post.assert_called_once()
        assert "Hola Mundo" in str(translated_json)


def test_process_pdf(sample_pdf, sample_json):
    """Test processing a single PDF file."""
    config = TranslationConfig(api_key="test-key", target_language="es")
    
    # Mock the necessary functions
    mock_provider = MagicMock()
    mock_provider.translate.return_value = {"translated": True}
    
    # Create a PatternManager to manage multiple patch contexts
    with patch('pdf_tools.processors.translator.PDFTranslator._create_provider', return_value=mock_provider), \
         patch('pdf_tools.processors.translator.PDFTranslator.extract_pdf_to_json', return_value=True), \
         patch('pdf_tools.processors.translator.PDFTranslator.generate_pdf_from_json', return_value=True), \
         patch('builtins.open', mock_open(read_data='{"test": "data"}')), \
         patch('json.load', return_value={"test": "data"}), \
         patch('json.dump'):
        
        translator = PDFTranslator(config)
        result = translator.process_pdf(sample_pdf)
        
        # Verify the file paths are returned
        assert "source_json" in result
        assert "target_json" in result
        assert "target_pdf" in result
        
        # Verify translate was called
        mock_provider.translate.assert_called_once()


def test_invalid_directory():
    """Test handling of invalid directory paths."""
    config = TranslationConfig(api_key="test-key", target_language="es")
    
    with patch('pdf_tools.processors.translator.OpenAIProvider'):
        translator = PDFTranslator(config)
        
        with pytest.raises(ValueError, match="Input directory does not exist"):
            translator.process_directory(
                Path("/nonexistent/path"),
                Path("/some/output")
            )
