# pdf_tools/generators/pdf_generator.py
"""Module for generating PDF files from structured JSON data."""

import json
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Try to import PyMuPDF for proper redaction
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: poetry add pymupdf")

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%m/%d/%y %H:%M:%S',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Map of font families to actual font files
FONT_MAPPING = {
    'Arial': {
        'normal': 'Helvetica',
        'bold': 'Helvetica-Bold',
        'italic': 'Helvetica-Oblique',
        'bolditalic': 'Helvetica-BoldOblique'
    },
    'Times': {
        'normal': 'Times-Roman',
        'bold': 'Times-Bold',
        'italic': 'Times-Italic',
        'bolditalic': 'Times-BoldItalic'
    },
    'Courier': {
        'normal': 'Courier',
        'bold': 'Courier-Bold',
        'italic': 'Courier-Oblique',
        'bolditalic': 'Courier-BoldOblique'
    },
    # Default mapping for unknown fonts
    'Unknown': {
        'normal': 'Helvetica',
        'bold': 'Helvetica-Bold',
        'italic': 'Helvetica-Oblique',
        'bolditalic': 'Helvetica-BoldOblique'
    }
}


def resolve_font(family: str, weight: str, style: str) -> str:
    """Resolve font family, weight and style to an actual font name."""
    # Normalize input
    family = family.split(',')[0].strip()
    weight_key = 'bold' if weight.lower() in ('bold', '700') else 'normal'
    style_key = 'italic' if style.lower() == 'italic' else 'normal'

    # Create combined key
    if weight_key == 'bold' and style_key == 'italic':
        type_key = 'bolditalic'
    elif weight_key == 'bold':
        type_key = 'bold'
    elif style_key == 'italic':
        type_key = 'italic'
    else:
        type_key = 'normal'

    # Get font mapping for family or default to Unknown
    family_mapping = FONT_MAPPING.get(family, FONT_MAPPING.get('Unknown', {'normal': 'Helvetica'}))

    # Return font name
    return family_mapping.get(type_key, family_mapping.get('normal', 'Helvetica'))


def find_original_pdf(json_path: Path) -> Optional[Path]:
    """Find the original PDF file based on the JSON path."""
    try:
        # Load the JSON to get the original filename
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_filename = data.get('metadata', {}).get('originalFileName')
        if not original_filename:
            logger.warning("Original filename not found in metadata")
            return None

        # Check if original file exists in the same directory
        potential_path = json_path.parent / original_filename
        if potential_path.exists():
            logger.info(f"Found original PDF: {potential_path}")
            return potential_path

        # If not, look for the non-translated JSON and derive the path
        if '_' in json_path.stem:  # Suggests a translated JSON like "file_spa.json"
            # Get the source JSON filename (without language code)
            parts = json_path.stem.split('_')
            base_name = '_'.join(parts[:-1])
            source_json = json_path.parent / f"{base_name}.json"

            if source_json.exists():
                # Get the path from the source JSON
                with open(source_json, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)

                source_filename = source_data.get('metadata', {}).get('originalFileName')
                if source_filename:
                    source_path = json_path.parent / source_filename
                    if source_path.exists():
                        logger.info(f"Found original PDF from source JSON: {source_path}")
                        return source_path

        # Look in the input directory derived from path
        # Assume we're in /output/path/file_spa.json, original might be in /input/path/file.pdf
        if '_' in json_path.stem and 'target' in json_path.parts:
            target_index = json_path.parts.index('target')

            if target_index > 0 and target_index < len(json_path.parts) - 1:
                # Get the parent directory name (one up from 'target')
                parent_dir = json_path.parts[target_index - 1]

                # Try to find a 'source' directory at the same level
                source_dir = Path(*json_path.parts[:target_index]) / 'source'
                if source_dir.exists() and source_dir.is_dir():
                    # Get the base filename without language code
                    parts = json_path.stem.split('_')
                    base_name = '_'.join(parts[:-1])

                    # Look for potential files
                    potential_files = list(source_dir.glob(f"{base_name}*"))
                    if potential_files:
                        for file in potential_files:
                            if file.suffix.lower() == '.pdf':
                                logger.info(f"Found original PDF in source directory: {file}")
                                return file

        logger.warning("Could not find original PDF file")
        return None

    except Exception as e:
        logger.error(f"Error finding original PDF: {str(e)}")
        return None  # Changed from False to None for consistency


def translate_pdf_with_redaction(original_path: Path, json_path: Path, output_path: Path) -> bool:
    """
    Translate a PDF by properly removing original text using redaction and adding translated text.

    This approach:
    1. Uses PyMuPDF to properly redact (remove) the original text
    2. Adds the translated text at the exact same positions
    3. Uses a direct binary write approach to avoid file replacement issues
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF is required for redaction. Install with: poetry add pymupdf")
        return False

    try:
        logger.info(f"Translating PDF with redaction: {original_path}")
        logger.debug(f"Output path: {output_path}")
        
        # Check if output path parent directories are writable
        if not os.access(output_path.parent, os.W_OK):
            logger.error(f"Output directory is not writable: {output_path.parent}")
            return False

        # Load the source and target JSON structures
        target_json_path = json_path

        # Get the source JSON path
        source_json_path = None
        if "_" in target_json_path.stem:
            parts = target_json_path.stem.split("_")
            base_name = "_".join(parts[:-1])
            source_json_path = target_json_path.parent / f"{base_name}.json"

        if not source_json_path or not source_json_path.exists():
            logger.error(f"Source JSON not found for: {target_json_path}")
            return False

        # Load source and target JSON
        try:
            with open(source_json_path, 'r', encoding='utf-8') as f:
                source_structure = json.load(f)

            with open(target_json_path, 'r', encoding='utf-8') as f:
                target_structure = json.load(f)
        except json.JSONDecodeError as je:
            logger.error(f"Invalid JSON format: {je}")
            return False
        except Exception as e:
            logger.error(f"Error loading JSON files: {str(e)}")
            return False

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_pdf_path = temp_dir_path / "temp.pdf"
            logger.debug(f"Using temporary directory: {temp_dir_path}")

            # Copy original PDF to work with
            try:
                shutil.copy2(original_path, temp_pdf_path)
                logger.debug(f"Successfully copied original PDF to temp location: {temp_pdf_path}")
            except Exception as e:
                logger.error(f"Error copying original PDF to temp location: {str(e)}")
                return False

            # Open PDF with PyMuPDF
            try:
                doc = fitz.open(str(temp_pdf_path))
                logger.debug(f"Successfully opened PDF with PyMuPDF, pages: {len(doc)}")
            except Exception as e:
                logger.error(f"Error opening PDF with PyMuPDF: {str(e)}")
                return False

            # Process each page
            try:
                for page_idx in range(len(doc)):
                    page_number = page_idx + 1
                    page = doc[page_idx]
                    logger.debug(f"Processing page {page_number}")

                    # Get source and target elements for this page
                    source_elements = []
                    for p in source_structure.get('pages', []):
                        if p.get('pageNumber', 1) == page_number:
                            source_elements = [e for e in p.get('elements', []) if e.get('type') == 'text']
                            break

                    target_elements = []
                    for p in target_structure.get('pages', []):
                        if p.get('pageNumber', 1) == page_number:
                            target_elements = [e for e in p.get('elements', []) if e.get('type') == 'text']
                            break

                    if not source_elements:
                        logger.warning(f"No source text elements found for page {page_number}")
                        continue

                    if not target_elements:
                        logger.warning(f"No target text elements found for page {page_number}")
                        continue

                    if len(source_elements) != len(target_elements):
                        logger.warning(f"Mismatch in number of text elements on page {page_number}: {len(source_elements)} source vs {len(target_elements)} target")

                    # Enhanced redaction approach
                    
                    # First pass: Create redaction annotations for all text
                    redaction_list = []
                    for element in source_elements:
                        bbox = element.get('bbox', [0, 0, 0, 0])
                        # Make sure the bbox is valid and has some area
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            # Create a slightly larger rectangle to ensure complete text removal
                            # Add padding of 2 points on all sides
                            rect = fitz.Rect(
                                bbox[0] - 2,
                                bbox[1] - 2,
                                bbox[2] + 2,
                                bbox[3] + 2
                            )
                            
                            # Use white fill for redactions to match background
                            annot = page.add_redact_annot(
                                rect,
                                fill=(1, 1, 1),  # White fill
                                text_color=(1, 1, 1),  # White text (invisible)
                                cross_out=False  # Don't draw a cross through the redacted area
                            )
                            
                            redaction_list.append({
                                "rect": rect,
                                "annot": annot,
                                "content": element.get('content', '')
                            })
                    
                    # Apply all redactions at once
                    logger.info(f"Applying {len(redaction_list)} redactions on page {page_number}")
                    page.apply_redactions()
                    
                    # Get all page contents to verify redaction
                    text_before_insert = page.get_text()
                    logger.debug(f"Page {page_number} text length after redaction: {len(text_before_insert)}")
                    
                    # Now add the translated text
                    for idx, element in enumerate(target_elements):
                        content = element.get('content', '')
                        bbox = element.get('bbox', [0, 0, 0, 0])
                        style = element.get('style', {})

                        # Get font properties
                        font_size = style.get('fontSize', 12)
                        font_family = style.get('fontFamily', 'Helvetica')
                        font_weight = style.get('fontWeight', 'normal')
                        
                        # Try to get the best matching font
                        font_name = resolve_font(font_family, font_weight, style.get('fontStyle', 'normal'))
                        
                        # Get color
                        color_str = style.get('color', '#000000')
                        if isinstance(color_str, str) and color_str.startswith('#'):
                            color_str = color_str[1:]
                            r = int(color_str[0:2], 16) / 255
                            g = int(color_str[2:4], 16) / 255
                            b = int(color_str[4:6], 16) / 255
                            text_color = (r, g, b)
                        else:
                            text_color = (0, 0, 0)

                        # Calculate position - adjust Y for font baseline
                        # PyMuPDF uses bottom-left origin for text insertion
                        # The bbox uses top-left origin, so we need to adjust for the font height
                        x = bbox[0]
                        y = bbox[1] + font_size  # Adjust Y for baseline
                        
                        # Create a text insertion point
                        text_point = fitz.Point(x, y)
                        
                        # Add the translated text using insert_text
                        try:
                            page.insert_text(
                                text_point,
                                content,
                                fontsize=font_size,
                                color=text_color,
                                render_mode=0  # 0 = fill text
                            )
                            logger.debug(f"Added text '{content}' at ({x}, {y})")
                        except Exception as e:
                            logger.error(f"Error inserting text '{content}': {str(e)}")

                    # Additional text cleanup (optional)
                    # Use a second redaction pass if needed
                    if True:  # Enable secondary redaction pass to ensure all original text is removed
                        # Check for any remnant text and redact again if needed
                        words = page.get_text("words")
                        for w in words:
                            # Skip our newly inserted text by comparing positions
                            is_inserted = False
                            for element in target_elements:
                                bbox = element.get('bbox', [0, 0, 0, 0])
                                # Check if this word overlaps with any of our translations
                                if (w[0] >= bbox[0] - 5 and w[2] <= bbox[2] + 5 and 
                                    w[1] >= bbox[1] - 5 and w[3] <= bbox[3] + 5):
                                    is_inserted = True
                                    break
                            
                            # If not our inserted text, redact it
                            if not is_inserted:
                                rect = fitz.Rect(w[:4])
                                # Only redact if the rectangle has area
                                if rect.get_area() > 0:
                                    annot = page.add_redact_annot(rect, fill=(1, 1, 1))
                        
                        # Apply second redaction
                        page.apply_redactions()
                
                # Save to a temporary file
                temp_output_path = temp_dir_path / "output.pdf"
                doc.save(
                    str(temp_output_path),
                    garbage=4,        # Maximum garbage collection
                    deflate=True,     # Compress streams
                    clean=True,       # Clean content streams
                    pretty=False,     # Don't prettify
                    linear=True       # Optimize for web
                )
                doc.close()
                
                # Now use a binary read/write approach to avoid file replacement issues
                try:
                    # Create parent directories if they don't exist
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Read the temporary file as binary
                    with open(temp_output_path, 'rb') as source_file:
                        pdf_data = source_file.read()
                    
                    # Write directly to the destination as binary
                    with open(output_path, 'wb') as dest_file:
                        dest_file.write(pdf_data)
                    
                    logger.info(f"Successfully created translated PDF: {output_path}")
                    return True
                    
                except PermissionError as pe:
                    logger.error(f"Permission error copying to output path: {str(pe)}")
                    # Try more drastic measures for Windows compatibility
                    try:
                        logger.warning("Trying Windows compatibility mode for file writing")
                        if os.path.exists(output_path):
                            try:
                                os.chmod(output_path, 0o666)  # Make it writable if exists
                            except:
                                pass
                        
                        # Try to write the file in chunks to avoid file locking issues
                        with open(temp_output_path, 'rb') as source_file:
                            with open(output_path, 'wb') as dest_file:
                                # Copy in chunks of 1MB
                                chunk_size = 1024 * 1024
                                while True:
                                    chunk = source_file.read(chunk_size)
                                    if not chunk:
                                        break
                                    dest_file.write(chunk)
                        
                        logger.info(f"Successfully created translated PDF using Windows compatibility mode: {output_path}")
                        return True
                    except Exception as e2:
                        logger.error(f"Error in Windows compatibility mode: {str(e2)}")
                        return False
                except Exception as e:
                    logger.error(f"Error writing output file: {str(e)}")
                    return False
                
            except Exception as e:
                logger.error(f"Error processing PDF pages: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False

    except Exception as e:
        logger.error(f"Error in redaction translation: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def generate_pdf_from_structure(json_path: Path, pdf_path: Path) -> bool:
    """Generate a PDF file from a structured JSON representation."""
    try:
        logger.info(f"Generating PDF from structure: {json_path}")

        # Find the original PDF
        original_pdf = find_original_pdf(json_path)

        if original_pdf and original_pdf.exists() and PYMUPDF_AVAILABLE:
            # Use the redaction approach (most reliable for preserving layout)
            logger.info(f"Using redaction approach with original PDF: {original_pdf}")
            return translate_pdf_with_redaction(original_pdf, json_path, pdf_path)

        # Fallback to basic text PDF if redaction approach fails
        logger.warning("Redaction approach not available, using basic approach")

        # Load the JSON structure
        with open(json_path, 'r', encoding='utf-8') as json_file:
            structure = json.load(json_file)

        # Create PDF canvas
        c = canvas.Canvas(str(pdf_path))

        # Process each page
        for page in structure.get('pages', []):
            page_width = page.get('width', letter[0])
            page_height = page.get('height', letter[1])
            c.setPageSize((page_width, page_height))

            # Process each element
            for element in page.get('elements', []):
                element_type = element.get('type')

                if element_type == 'text':
                    # Extract text properties
                    content = element.get('content', '')
                    bbox = element.get('bbox', [0, 0, 0, 0])
                    style = element.get('style', {})

                    # Get font properties
                    font_family = style.get('fontFamily', 'Helvetica')
                    font_size = style.get('fontSize', 12)
                    font_weight = style.get('fontWeight', 'normal')
                    font_style = style.get('fontStyle', 'normal')
                    color = style.get('color', '#000000')

                    # Set font
                    font_name = resolve_font(font_family, font_weight, font_style)
                    c.setFont(font_name, font_size)

                    # Set color (remove # if present)
                    if isinstance(color, str) and color.startswith('#'):
                        color = color[1:]
                    c.setFillColor(HexColor(f"#{color}") if isinstance(color, str) else black)

                    # Calculate text position
                    x = bbox[0]
                    y = page_height - bbox[1] - font_size  # Adjust for ReportLab's coordinate system

                    # Handle text alignment
                    alignment = style.get('alignment', 'left')
                    width = bbox[2] - bbox[0]

                    # Draw the text with proper alignment
                    if alignment == 'center':
                        c.drawCentredString(x + width/2, y, content)
                    elif alignment == 'right':
                        c.drawRightString(x + width, y, content)
                    else:  # left or default
                        c.drawString(x, y, content)

            # Add the page
            c.showPage()

        # Save the PDF
        c.save()
        logger.info(f"Successfully generated PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


# Alias the function to maintain backward compatibility
generate_pdf_from_json = generate_pdf_from_structure


def main():
    """CLI entry point for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate PDF from structured JSON")
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("output_pdf", help="Path to save the output PDF")
    args = parser.parse_args()

    generate_pdf_from_structure(Path(args.input_json), Path(args.output_pdf))


if __name__ == "__main__":
    main()
