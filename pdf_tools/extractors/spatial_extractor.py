# pdf_tools/extractors/spatial_extractor.py
"""Module for extracting spatial information from PDF files."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Use multiple PDF extraction libraries for better coverage
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import (
    LTPage, LTTextContainer, LTChar, LTTextLine, LTTextBox, 
    LTFigure, LTImage, LTRect, LTLine
)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFStream, PDFObjRef
from pdfminer.utils import matrix2str, Rect

# Alternative extraction method
import pypdf

# For image detection
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%m/%d/%y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def debug_pdf_structure(pdf_path: Path) -> None:
    """Debug PDF structure to help identify extraction issues."""
    logger.info(f"Debugging PDF structure for {pdf_path}")
    
    # Method 1: Use pypdf to analyze document
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            logger.info(f"PDF has {len(reader.pages)} pages")
            
            # Check if document is encrypted
            logger.info(f"Document is {'encrypted' if reader.is_encrypted else 'not encrypted'}")
            
            # Check if document is form
            has_forms = False
            for page in reader.pages:
                if '/Annots' in page:
                    has_forms = True
                    break
            logger.info(f"Document {'has forms' if has_forms else 'does not have forms'}")
            
            # Check metadata
            metadata = reader.metadata
            if metadata:
                logger.info(f"Document metadata: {metadata}")
            
            # Check first page text (direct extraction)
            try:
                text = reader.pages[0].extract_text()
                text_preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"First page text preview (pypdf): {text_preview}")
                logger.info(f"Text length from first page: {len(text)}")
            except Exception as e:
                logger.error(f"Error extracting text with pypdf: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error analyzing PDF with pypdf: {str(e)}")
    
    # Method 2: Use pdfminer for detailed structure
    try:
        try:
            text = extract_text(str(pdf_path), page_numbers=[0])
            text_preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"First page text preview (pdfminer direct): {text_preview}")
            logger.info(f"Text length from first page (direct): {len(text)}")
        except Exception as e:
            logger.error(f"Error with direct pdfminer extraction: {str(e)}")
        
        with open(pdf_path, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            
            # Check document structure
            try:
                for xref in document.xrefs:
                    if hasattr(xref, 'get_objects'):
                        logger.info(f"PDF has {len(xref.get_objects())} objects")
            except Exception as e:
                logger.error(f"Error getting objects: {str(e)}")
            
            # Extract and log first page elements
            try:
                for page_layout in extract_pages(str(pdf_path), maxpages=1):
                    element_count = {
                        'LTTextBox': 0,
                        'LTTextLine': 0,
                        'LTChar': 0,
                        'LTFigure': 0,
                        'LTImage': 0,
                        'Other': 0
                    }
                    
                    def count_elements(element):
                        element_type = type(element).__name__
                        if element_type in element_count:
                            element_count[element_type] += 1
                        else:
                            element_count['Other'] += 1
                        
                        # Recursive count for containers
                        if hasattr(element, '_objs'):
                            for obj in element._objs:
                                count_elements(obj)
                    
                    # Count all elements
                    count_elements(page_layout)
                    logger.info(f"First page element counts: {element_count}")
                    
                    # Check if page has any text content
                    text_content = ""
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            text_content += element.get_text()
                    
                    text_preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
                    logger.info(f"First page text preview (pdfminer layout): {text_preview}")
                    logger.info(f"Text length from first page (layout): {len(text_content)}")
            except Exception as e:
                logger.error(f"Error processing page layout: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error analyzing PDF with pdfminer: {str(e)}")


def extract_font_info(obj: Any) -> Dict[str, Any]:
    """Extract font information from a PDF object."""
    font_info = {}
    
    if isinstance(obj, dict):
        if 'BaseFont' in obj:
            font_name = obj['BaseFont']
            if isinstance(font_name, bytes):
                font_name = font_name.decode('utf-8', errors='replace')
            if font_name.startswith('/'):
                font_name = font_name[1:]
            font_info['fontFamily'] = font_name
        
        if 'FontDescriptor' in obj:
            font_descriptor = obj['FontDescriptor']
            if isinstance(font_descriptor, PDFObjRef):
                font_descriptor = font_descriptor.resolve()
            
            if isinstance(font_descriptor, dict):
                if 'FontWeight' in font_descriptor:
                    font_info['fontWeight'] = font_descriptor['FontWeight']
                
                if 'Flags' in font_descriptor:
                    flags = font_descriptor['Flags']
                    # Check if bit 3 is set (italic)
                    if flags & (1 << 6):
                        font_info['fontStyle'] = 'italic'
                    else:
                        font_info['fontStyle'] = 'normal'
    
    return font_info


def extract_with_pdfminer(pdf_path: Path) -> Dict[str, Any]:
    """Extract PDF structure using PDFMiner."""
    pdf_structure = {
        "metadata": {
            "originalFileName": pdf_path.name,
            "language": "auto",
            "creationDate": None,
        },
        "pages": []
    }
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            
            # Extract document metadata
            if document.info and 'CreationDate' in document.info[0]:
                creation_date = document.info[0]['CreationDate']
                if isinstance(creation_date, bytes):
                    creation_date = creation_date.decode('utf-8', errors='replace')
                pdf_structure['metadata']['creationDate'] = creation_date
            
            # Process each page
            for page_idx, page_layout in enumerate(extract_pages(str(pdf_path))):
                logger.info(f"Processing page {page_idx + 1} with PDFMiner")
                
                # Get page dimensions
                page_width = page_layout.width
                page_height = page_layout.height
                
                page_structure = {
                    "pageNumber": page_idx + 1,
                    "width": page_width,
                    "height": page_height,
                    "elements": []
                }
                
                # Extract text elements with spatial information
                element_id = 1
                
                # Process all elements
                def process_element(element, page_height):
                    nonlocal element_id
                    
                    if isinstance(element, LTTextBox):
                        for text_line in element:
                            if isinstance(text_line, LTTextLine):
                                process_text_line(text_line, page_height, element_id)
                                element_id += 1
                    elif isinstance(element, LTTextLine):
                        process_text_line(element, page_height, element_id)
                        element_id += 1
                    elif isinstance(element, LTFigure):
                        # Process figures (might contain text)
                        for child in element:
                            process_element(child, page_height)
                    elif isinstance(element, LTImage):
                        # Process images
                        bbox = [
                            element.x0, 
                            page_height - element.y1,
                            element.x1, 
                            page_height - element.y0
                        ]
                        image_element = {
                            "id": f"image-{element_id:03d}",
                            "type": "image",
                            "bbox": bbox,
                        }
                        page_structure["elements"].append(image_element)
                        element_id += 1
                
                def process_text_line(text_line, page_height, current_id):
                    text_content = text_line.get_text().strip()
                    if not text_content:
                        return
                    
                    # Get bounding box
                    bbox = [
                        text_line.x0, 
                        page_height - text_line.y1,
                        text_line.x1, 
                        page_height - text_line.y0
                    ]
                    
                    # Collect style information
                    style = {
                        "fontFamily": "Unknown",
                        "fontSize": 0,
                        "fontWeight": "normal",
                        "fontStyle": "normal",
                        "color": "#000000",  # Default black
                        "alignment": "left",  # Default alignment
                        "lineHeight": 1.2,   # Default line height
                    }
                    
                    # Calculate average font size and collect font information
                    font_sizes = []
                    for char in text_line:
                        if isinstance(char, LTChar):
                            font_sizes.append(char.size)
                            
                            # Get font information
                            if hasattr(char, 'fontname') and char.fontname:
                                style["fontFamily"] = char.fontname.split('+')[-1]
                    
                    # Set the font size to the average
                    if font_sizes:
                        style["fontSize"] = sum(font_sizes) / len(font_sizes)
                    
                    # Create the text element
                    text_element = {
                        "id": f"text-{current_id:03d}",
                        "type": "text",
                        "content": text_content,
                        "bbox": bbox,
                        "style": style,
                        "baselineY": page_height - text_line.y0  # Baseline in top-left origin
                    }
                    
                    page_structure["elements"].append(text_element)
                
                # Process all elements on the page
                for element in page_layout:
                    process_element(element, page_height)
                
                # Add the page to the structure
                pdf_structure["pages"].append(page_structure)
                
                # Log extraction results
                logger.info(f"Extracted {len(page_structure['elements'])} elements from page {page_idx + 1}")
                
    except Exception as e:
        logger.error(f"Error extracting with PDFMiner: {str(e)}")
        raise
        
    return pdf_structure


def split_text_into_lines(text: str) -> List[str]:
    """Split text into lines and remove excess whitespace."""
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def extract_with_pypdf(pdf_path: Path) -> Dict[str, Any]:
    """Extract PDF structure using PyPDF."""
    pdf_structure = {
        "metadata": {
            "originalFileName": pdf_path.name,
            "language": "auto",
            "creationDate": None,
        },
        "pages": []
    }
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            
            # Extract document metadata
            if reader.metadata:
                if reader.metadata.get('/CreationDate'):
                    pdf_structure['metadata']['creationDate'] = reader.metadata.get('/CreationDate')
                elif reader.metadata.get('CreationDate'):  # Different format
                    pdf_structure['metadata']['creationDate'] = reader.metadata.get('CreationDate')
            
            # Process each page
            for page_idx, page in enumerate(reader.pages):
                logger.info(f"Processing page {page_idx + 1} with PyPDF")
                
                # Get page dimensions
                media_box = page.mediabox
                page_width = float(media_box.width)
                page_height = float(media_box.height)
                
                page_structure = {
                    "pageNumber": page_idx + 1,
                    "width": page_width,
                    "height": page_height,
                    "elements": []
                }
                
                # Extract text elements
                text = page.extract_text()
                
                if text:
                    logger.info(f"PyPDF extracted {len(text)} characters of text")
                    lines = split_text_into_lines(text)
                    logger.info(f"Text split into {len(lines)} lines")
                    
                    # Create elements for each line of text
                    for i, line in enumerate(lines):
                        # We don't have precise positioning from PyPDF,
                        # so we'll create estimated positions based on line number
                        element_id = i + 1
                        line_height = 14  # Approximate line height
                        y_pos = i * line_height
                        
                        # Determine if this might be a heading based on all caps
                        is_heading = line.isupper() or (
                            len(line) < 50 and sum(1 for c in line if c.isupper()) / len(line) > 0.7
                        )
                        
                        font_size = 14 if is_heading else 12
                        font_weight = "bold" if is_heading else "normal"
                        
                        # Estimate width based on content length
                        char_width = 7  # Approximate character width in points
                        width = min(len(line) * char_width, page_width - 100)  # With some margin
                        
                        # Left margin
                        x_margin = 50
                        
                        text_element = {
                            "id": f"text-{element_id:03d}",
                            "type": "text",
                            "content": line,
                            "bbox": [x_margin, y_pos, x_margin + width, y_pos + line_height],
                            "style": {
                                "fontFamily": "Arial",
                                "fontSize": font_size,
                                "fontWeight": font_weight,
                                "fontStyle": "normal",
                                "color": "#000000",
                                "alignment": "left",
                                "lineHeight": 1.2,
                            },
                            "baselineY": y_pos + line_height * 0.8  # Approximate baseline
                        }
                        page_structure["elements"].append(text_element)
                
                # Check if we found text elements
                if not page_structure["elements"]:
                    # Add a full-page image as fallback
                    page_structure["elements"].append({
                        "id": "image-001",
                        "type": "image",
                        "bbox": [0, 0, page_width, page_height],
                    })
                
                # Add the page to the structure
                pdf_structure["pages"].append(page_structure)
                
                # Log extraction results
                logger.info(f"Extracted {len(page_structure['elements'])} elements from page {page_idx + 1}")
                
    except Exception as e:
        logger.error(f"Error extracting with PyPDF: {str(e)}")
    
    return pdf_structure


def extract_pdf_structure(pdf_path: Path, output_json_path: Path) -> bool:
    """Extract spatial information from a PDF file into a structured JSON."""
    try:
        logger.info(f"Extracting spatial information from {pdf_path}")
        
        # Run debug information first
        debug_pdf_structure(pdf_path)
        
        # Try extraction with PDFMiner first
        try:
            pdf_structure = extract_with_pdfminer(pdf_path)
            
            # Check if any elements were extracted
            total_elements = sum(len(page.get('elements', [])) for page in pdf_structure.get('pages', []))
            logger.info(f"Total elements extracted with PDFMiner: {total_elements}")
            
            # If no text elements found, try with PyPDF
            text_elements = 0
            for page in pdf_structure.get('pages', []):
                for element in page.get('elements', []):
                    if element.get('type') == 'text':
                        text_elements += 1
            
            if text_elements == 0:
                logger.info("No text elements found with PDFMiner, trying PyPDF")
                pdf_structure = extract_with_pypdf(pdf_path)
                total_elements = sum(len(page.get('elements', [])) for page in pdf_structure.get('pages', []))
                text_elements = 0
                for page in pdf_structure.get('pages', []):
                    for element in page.get('elements', []):
                        if element.get('type') == 'text':
                            text_elements += 1
                logger.info(f"Total elements extracted with PyPDF: {total_elements} (text elements: {text_elements})")
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {str(e)}")
            logger.info("Falling back to PyPDF extraction")
            pdf_structure = extract_with_pypdf(pdf_path)
            total_elements = sum(len(page.get('elements', [])) for page in pdf_structure.get('pages', []))
            logger.info(f"Total elements extracted with PyPDF: {total_elements}")
        
        # Final check - do we have at least one text element?
        text_elements = 0
        for page in pdf_structure.get('pages', []):
            for element in page.get('elements', []):
                if element.get('type') == 'text':
                    text_elements += 1
        
        if text_elements == 0:
            logger.warning("No text elements found in the PDF! Final attempt with direct text extraction")
            # Last resort - direct text extraction and simple positioning
            try:
                with open(pdf_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page_idx, page in enumerate(reader.pages):
                        page_obj = pdf_structure['pages'][page_idx]
                        text = page.extract_text()
                        
                        if text:
                            logger.info(f"Direct extraction found {len(text)} characters")
                            lines = split_text_into_lines(text)
                            logger.info(f"Text split into {len(lines)} lines")
                            
                            # Clear existing elements (probably just images)
                            page_obj['elements'] = []
                            
                            # Create elements for each line of text
                            for i, line in enumerate(lines):
                                element_id = i + 1
                                line_height = 14
                                y_pos = i * line_height
                                
                                is_heading = line.isupper() or (
                                    len(line) < 50 and sum(1 for c in line if c.isupper()) / len(line) > 0.7
                                )
                                
                                font_size = 14 if is_heading else 12
                                font_weight = "bold" if is_heading else "normal"
                                
                                char_width = 7
                                width = min(len(line) * char_width, page_obj['width'] - 100)
                                x_margin = 50
                                
                                text_element = {
                                    "id": f"text-{element_id:03d}",
                                    "type": "text",
                                    "content": line,
                                    "bbox": [x_margin, y_pos, x_margin + width, y_pos + line_height],
                                    "style": {
                                        "fontFamily": "Arial",
                                        "fontSize": font_size,
                                        "fontWeight": font_weight,
                                        "fontStyle": "normal",
                                        "color": "#000000",
                                        "alignment": "left",
                                        "lineHeight": 1.2,
                                    },
                                    "baselineY": y_pos + line_height * 0.8
                                }
                                page_obj['elements'].append(text_element)
            except Exception as e:
                logger.error(f"Final text extraction attempt failed: {str(e)}")
        
        # Write the JSON structure to file
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(pdf_structure, json_file, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully extracted structure to {output_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting PDF structure: {str(e)}")
        return False


def main():
    """CLI entry point for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract spatial information from PDF")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("output_json", help="Path to save the output JSON")
    args = parser.parse_args()
    
    extract_pdf_structure(Path(args.input_pdf), Path(args.output_json))


if __name__ == "__main__":
    main()
