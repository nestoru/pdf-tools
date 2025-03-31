# pdf_tools/cli.py
"""Command-line interface for PDF tools."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pdf_tools.processors.ocr import OCRConfig, PDFOCRProcessor

# Initialize typer app and console
app = typer.Typer(
    help="PDF processing tools including OCR capabilities",
    rich_markup_mode="rich",
)
console = Console()

def setup_logging(verbose: bool):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def display_results_table(results):
    """Display processing results in a formatted table."""
    table = Table(title="Processing Results")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Files Processed", str(results.processed_count))
    table.add_row("Files Skipped", str(results.skipped_count))
    table.add_row("Processing Time", f"{results.total_processing_time:.2f}s")

    if results.errors:
        table.add_row("Errors", str(len(results.errors)))

    console.print(table)

@app.command(name="simple")
def simple_ocr(
    input_dir: str = typer.Argument(..., help="Input directory path"),
    output_dir: str = typer.Argument(..., help="Output directory path")
):
    """
    Simple OCR command matching the original format: ocr_pdf.py <input_dir> <output_dir>

    This provides a straightforward way to process PDFs with default settings.
    """
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.is_dir():
            console.print(f"[red]Error: Input directory does not exist: {input_dir}[/]")
            raise typer.Exit(1)

        # Ensure the output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check for zero-sized files in the output directory before processing
        processor = PDFOCRProcessor()
        
        # Check for any existing empty PDF files in the output directory
        empty_files_found = False
        for pdf_path in output_path.rglob("*.pdf"):
            if pdf_path.exists() and pdf_path.stat().st_size == 0:
                console.print(f"[yellow]Warning: Found empty PDF file: {pdf_path}[/]")
                try:
                    pdf_path.unlink()
                    console.print(f"[green]Removed empty file: {pdf_path}[/]")
                except Exception as e:
                    console.print(f"[red]Failed to remove empty file {pdf_path}: {str(e)}[/]")
                empty_files_found = True
                
        if empty_files_found:
            console.print("[yellow]Cleaned up empty PDF files from output directory[/]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Processing PDF files...", total=None)
            results = processor.process_directory(input_path, output_path)

        display_results_table(results)

        if results.errors:
            console.print("\n[bold red]Errors occurred:[/]")
            for error in results.errors:
                console.print(f"[red]- {error}[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {str(e)}")
        raise typer.Exit(1)

@app.command(name="ocr")
def advanced_ocr(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDF files to process",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory to save processed PDF files",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    languages: Optional[List[str]] = typer.Option(
        None,
        "--lang",
        "-l",
        help="Languages to use for OCR (e.g., -l en -l fr)",
    ),
    dpi: int = typer.Option(
        300,
        "--dpi",
        "-d",
        help="DPI for PDF to image conversion (higher means better quality but slower)",
        min=72,
        max=600,
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR on every page, even if text is detected",
    ),
    skip_text: bool = typer.Option(
        False,
        "--skip-text",
        help="Skip OCR on pages that already contain text",
    ),
    no_rotation: bool = typer.Option(
        False,
        "--no-rotation",
        help="Disable automatic rotation detection and correction",
    ),
    no_cleaning: bool = typer.Option(
        False,
        "--no-cleaning",
        help="Disable cleaning of visual noise from scanned images",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Process PDF files with OCR capabilities with advanced options.

    This command provides full control over OCR settings and processing options.
    """
    setup_logging(verbose)

    try:
        config = OCRConfig(
            languages=languages or ['eng'],
            dpi=dpi,
            force_ocr=force_ocr,
            skip_text=skip_text,
            rotate_pages=not no_rotation,
            clean=not no_cleaning,
        )

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for zero-sized files in the output directory before processing
        empty_files_found = False
        for pdf_path in output_dir.rglob("*.pdf"):
            if pdf_path.exists() and pdf_path.stat().st_size == 0:
                console.print(f"[yellow]Warning: Found empty PDF file: {pdf_path}[/]")
                try:
                    pdf_path.unlink()
                    console.print(f"[green]Removed empty file: {pdf_path}[/]")
                except Exception as e:
                    console.print(f"[red]Failed to remove empty file {pdf_path}: {str(e)}[/]")
                empty_files_found = True
                
        if empty_files_found:
            console.print("[yellow]Cleaned up empty PDF files from output directory[/]")

        processor = PDFOCRProcessor(config=config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Processing PDF files...", total=None)
            results = processor.process_directory(input_dir, output_dir)

        display_results_table(results)

        if results.errors:
            console.print("\n[bold red]Errors occurred:[/]")
            for error in results.errors:
                console.print(f"[red]- {error}[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {str(e)}")
        raise typer.Exit(1)

def main():
    """Entry point for direct script execution."""
    if len(sys.argv) == 3:
        # If exactly two arguments are provided, use the simple command
        simple_ocr(sys.argv[1], sys.argv[2])
    else:
        # Otherwise, use the Typer CLI
        app()

if __name__ == "__main__":
    main()
