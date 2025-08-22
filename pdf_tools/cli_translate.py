# pdf_tools/cli_translate.py
"""Translation command for PDF tools."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table

from pdf_tools.processors.translator import TranslationConfig, PDFTranslator, DirectoryProcessResult

# Initialize console
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
    """Display translation results in a formatted table."""
    table = Table(title="Translation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files Processed", str(results.processed_count))
    table.add_row("Files Skipped", str(results.skipped_count))
    table.add_row("Processing Time", f"{results.total_processing_time:.2f}s")
    if results.errors:
        table.add_row("Errors", str(len(results.errors)))
    console.print(table)


def translate_command(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDF files to translate",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory to save translated PDF files",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    ml_provider: str = typer.Option(
        ...,  # Make mandatory
        "--ml-provider",
        help="ML provider (openai, anthropic)",
    ),
    ml_model: str = typer.Option(
        ...,  # Make mandatory
        "--ml-model",
        help="ML model to use",
    ),
    target_lang: str = typer.Option(
        ...,
        "--target-lang",
        "-l",
        help="Target language code (e.g., spa, fra, deu)",
    ),
    regenerate_target: bool = typer.Option(
        False,
        "--regenerate-target",
        help="Regenerate PDFs from existing PS files without translating again",
    ),
    keep_postscript: bool = typer.Option(
        False,
        "--keep-postscript",
        help="Save PostScript files for debugging and diffing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Translate PDF documents while preserving layout.

    This command directly manipulates PDF content streams to translate the
    text while maintaining the original layout and formatting. It uses an LLM
    to understand and modify the PostScript content of the PDF.

    The translation process uses timestamp-based processing to avoid redundant
    operations. Files will only be processed if they've been modified since
    the last translation or if no translation exists.

    With --keep-postscript, it will save the extracted and translated PostScript
    files alongside the PDFs for debugging and diffing.

    With --regenerate-target, it will use existing translated PostScript files
    if available to regenerate PDFs without doing a fresh translation.
    """
    start_time = datetime.now()
    setup_logging(verbose)
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        config_data = {}
        try:
            with open(config, "r") as f:
                config_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading config: {str(e)}[/bold red]")
            raise typer.Exit(1)
            
        # Add command line options to config
        config_data.update({
            'ml_provider': ml_provider,
            'ml_model': ml_model,
            'target_language': target_lang,
            'keep_postscript': keep_postscript
        })
        
        # Create translation configuration
        config_obj = TranslationConfig(**config_data)

        # Create translator
        translator = PDFTranslator(config=config_obj)

        # Find PDF files to process
        pdf_files = translator.find_pdf_files(input_dir)
        console.print(f"Found [bold]{len(pdf_files)}[/] PDF files to process")

        if not pdf_files:
            console.print("[yellow]No PDF files found to process[/]")
            raise typer.Exit(0)

        # Process with more detailed progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            # Create the overall progress task
            overall_task = progress.add_task("Processing PDF files...", total=len(pdf_files))
            current_file_task = progress.add_task("Current file", total=3, visible=False)

            # Initialize counters
            processed_count = 0
            skipped_count = 0
            errors = []

            for pdf_idx, input_path in enumerate(pdf_files, 1):
                # Update overall progress
                progress.update(overall_task, description=f"Processing PDF {pdf_idx}/{len(pdf_files)}")

                # Skip input files with zero size
                if input_path.stat().st_size == 0:
                    progress.console.print(f"[yellow]Skipping empty input file: {input_path.name}[/]")
                    continue

                # Calculate the relative path and construct the output path
                relative_path = input_path.relative_to(input_dir)
                base_name = relative_path.stem
                output_subdir = output_dir / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Output PDF path
                target_pdf_path = output_subdir / f"{base_name}_{config_obj.target_language}.pdf"

                # Reset and show current file task
                progress.update(current_file_task, completed=0, visible=True)

                try:
                    # Check if processing is needed
                    if regenerate_target or translator.should_process(input_path, target_pdf_path):
                        # Process the PDF
                        progress.update(current_file_task,
                                      description=f"Processing PDF: {input_path.name}",
                                      completed=1)
                        
                        # Process the file
                        if translator.process_pdf_file(
                            input_path=input_path,
                            output_path=target_pdf_path,
                            source_lang="en",  # Default source language
                            regenerate=regenerate_target
                        ):
                            processed_count += 1
                            progress.update(current_file_task, completed=3)
                            progress.console.print(f"[green]Successfully processed: {target_pdf_path.name}[/]")
                        else:
                            raise Exception(f"Failed to process PDF")
                    else:
                        progress.console.print(f"[cyan]Skipping {input_path.name} - output is newer than input[/]")
                        skipped_count += 1
                        
                except Exception as e:
                    error_msg = f"Error processing {input_path.name}: {str(e)}"
                    progress.console.print(f"[red]{error_msg}[/]")
                    errors.append(error_msg)

                # Hide the current file task between files
                progress.update(current_file_task, visible=False)

                # Update overall progress
                progress.update(overall_task, advance=1)

            # Complete the overall task
            progress.update(overall_task,
                            description=f"Completed processing {processed_count} files",
                            completed=len(pdf_files))

        # Create result object
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        results = DirectoryProcessResult(
            processed_count=processed_count,
            skipped_count=skipped_count,
            errors=errors,
            total_processing_time=processing_time
        )

        display_results_table(results)

        if results.errors:
            console.print("\n[bold red]Errors occurred:[/]")
            for error in results.errors:
                console.print(f"[red]- {error}[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(translate_command)
