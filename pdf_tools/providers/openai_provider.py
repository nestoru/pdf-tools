# pdf_tools/providers/openai_provider.py
"""OpenAI provider implementation for PDF translation."""

import json
import logging
import os
from typing import Dict, Any, Optional

from pdf_tools.processors.translator import MLTranslationProvider, TranslationConfig

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%m/%d/%y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class OpenAIProvider(MLTranslationProvider):
    """OpenAI implementation for translation."""

    def __init__(self, config: TranslationConfig):
        """Initialize the OpenAI provider."""
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")
            
        self.model = config.ml_model

        # Validate model existence
        self.supported_models = [
            'gpt-4-turbo', 'gpt-4', 'gpt-4-32k', 'gpt-4-1106-preview',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'
        ]
        if self.model not in self.supported_models:
            logger.warning(
                f"Model {self.model} is not in the list of tested models: "
                f"{', '.join(self.supported_models)}"
            )

    def translate_content_stream(self, content_stream: str, 
                               source_lang: str, target_lang: str) -> str:
        """
        Translate a PDF content stream.
        
        Args:
            content_stream: PDF content stream to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated content stream
        """
        # Create prompt for content stream translation
        prompt = self._get_content_stream_prompt(source_lang, target_lang)
        
        try:
            # Get translation from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"```\n{content_stream}\n```"}
                ],
                temperature=0.1,
                max_tokens=8192
            )
            
            # Extract translated content
            translated_content = response.choices[0].message.content
            
            # Extract just the content stream from the response
            # (remove any explanations or markers the AI might add)
            if "```" in translated_content:
                # Extract content between code markers if present
                parts = translated_content.split("```")
                for part in parts:
                    if part.strip() and not part.lower().startswith("postscript") and not part.lower().startswith("pdf"):
                        translated_content = part.strip()
                        break
            
            return translated_content
            
        except Exception as e:
            logger.error(f"Error in OpenAI translation: {str(e)}")
            raise

    def _get_current_date_iso(self) -> str:
        """Get current date in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _get_content_stream_prompt(self, source_lang: str, target_lang: str) -> str:
        """
        Get the prompt for content stream translation.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Prompt for content stream translation
        """
        return f"""
# PDF Content Stream Translation Expert

You are an expert in PDF and PostScript internals. You are given a PDF content stream
that contains text in {source_lang}. Your task is to:

1. Identify all text elements in the content stream, which are typically enclosed in 
   parentheses ( ) or angle brackets < > after Tj, TJ, or other text operators.
   
2. Translate ONLY the text content from {source_lang} to {target_lang}.

3. Preserve ALL PostScript operators, positions, font references, and graphics.
   Do not modify any operators or parameters except the text content itself.

4. Pay attention to text positioning. The PDF uses a coordinate system where elements 
   are positioned using operators like Td, TD, Tm. Maintain these positions exactly.

5. Preserve text styles including font selection, size, and formatting.

6. When translating text in TJ arrays, maintain the positioning adjustments between
   characters (the numbers between text segments).

7. Return the complete modified content stream with translations replacing the
   original text.

## Key PostScript text operators to understand:

- BT/ET: Begin/End text object
- Tf: Set font and size
- Td/TD/Tm: Set text position
- Tj: Show a text string
- TJ: Show text with positioning adjustments
- ': Move to next line and show text
- ": Set word and character spacing, move to next line, and show text

For TJ arrays, remember these are arrays that mix strings with numbers, like:
[(T) 120 (ext)] TJ
Where the numbers are horizontal adjustments between characters.

## Examples:

Original: (Hello world) Tj
Translated: ({target_lang} equivalent of "Hello world") Tj

Original: [(T) -120 (ext)] TJ
Translated: [({target_lang} first char) -120 ({target_lang} remaining chars)] TJ

## Return Format:

Return ONLY the modified content stream with all syntax preserved and only text translated.
Ensure all operators, positioning, and styles remain identical to the original.
Do NOT add any explanations or commentary - just the raw translated content stream.
"""

    def translate(self, source_json: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """Legacy method for JSON-based translation (kept for backward compatibility)."""
        logger.warning("Using legacy JSON-based translation method")
        source_language = source_json.get('metadata', {}).get('language', 'en')

        # Create a translation prompt
        prompt = """
# PDF Translation System

You are a specialized PDF translation system tasked with translating a document while preserving its layout integrity.

## Your Task

Translate the JSON representation of a PDF page from {source_language} to {target_language}, maintaining the spatial layout of all elements.

## Translation Requirements

1. Translate all text content accurately and fluently
2. NEVER modify the font family of any element
3. Adjust ONLY these properties to handle text expansion/contraction:
   - Font size (can be reduced up to 20% or increased up to 10%)
   - Character spacing (can be adjusted between -10% and +10%)
   - Word spacing (as a last resort)

## Important Rules

- Maintain the meaning and tone of the original text
- Preserve formatting (bold, italic, etc.) in appropriate places in the translation
- Adapt numbers, dates, and units according to {target_language} conventions
- Handle idioms and cultural references appropriately
- NEVER change the bounding box coordinates
- NEVER change the font family

## Output Format

Return a valid JSON object with exactly the same structure as the input, containing:
- The translated "content" for each text element
- Adjusted "style" properties as needed (fontSize, letterSpacing, wordSpacing)
- All other properties exactly as provided in the input
"""
        prompt = prompt.replace("{source_language}", source_language)
        prompt = prompt.replace("{target_language}", target_language)

        # Translated JSON with metadata
        translated_json = {
            **source_json,
            "metadata": {
                **source_json.get('metadata', {}),
                "sourceLanguage": source_language,
                "targetLanguage": target_language,
                "translationDate": self._get_current_date_iso()
            },
            "pages": []
        }

        # Process each page
        for page_idx, page in enumerate(source_json.get('pages', [])):
            logger.info(f"Translating page {page_idx + 1}/{len(source_json.get('pages', []))}")

            try:
                # Prepare the API request for this page
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(page)}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                # Extract the translated page
                translated_page_content = response.choices[0].message.content
                translated_page = json.loads(translated_page_content)

                # Add to the translated document
                translated_json['pages'].append(translated_page)

            except Exception as e:
                logger.error(f"Error translating page {page_idx + 1}: {str(e)}")

                # In case of error, add the original page without translation
                translated_json['pages'].append(page)

                # Record the error
                if 'translation_errors' not in translated_json:
                    translated_json['translation_errors'] = []

                translated_json['translation_errors'].append({
                    'pageNumber': page_idx + 1,
                    'error': str(e)
                })

        return translated_json
