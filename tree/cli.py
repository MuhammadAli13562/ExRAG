"""
CLI interface for the tree building pipeline.
"""
import json
from pathlib import Path
from typing import Optional, Literal
import typer
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tree.pipeline import build_tree
from tree.builder import print_tree
from common.logging_config import get_logger

logger = get_logger("exrag.tree.cli")

app = typer.Typer(
    help="Build hierarchical tree structures from Markdown files",
    add_completion=False,
)


@app.command("build")
def cli_build(
    md_file: Path = typer.Argument(..., help="Path to markdown file", exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path (auto-generated if not provided)"),
    mode: Literal["simple", "full"] = typer.Option("full", "--mode", "-m", help="Build mode"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if output exists"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """
    Build tree structure from a markdown file.
    
    Example:
        python -m tree build md/biology.md
        python -m tree build md/physics.md --output results/custom.json --mode simple
    """
    # Setup logging
    if verbose:
        logger.setLevel("DEBUG")
    
    try:
        result = build_tree(
            md_path=md_file,
            output_path=output,
            mode=mode,
            force=force,
        )
        
        if json_output:
            print(result.model_dump_json(indent=2))
        else:
            if result.skipped:
                typer.secho(f"✓ Skipped (unchanged): {result.output_path}", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"✓ Built successfully: {result.output_path}", fg=typer.colors.GREEN)
            typer.echo(f"  Document: {result.doc_name}")
            typer.echo(f"  Total nodes: {result.total_nodes}")
            typer.echo(f"  Root nodes: {result.root_nodes}")
            typer.echo(f"  Max depth: {result.max_depth}")
            typer.echo(f"  Content hash: {result.content_hash}")
        
        raise typer.Exit(0)
        
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("print")
def cli_print(
    json_file: Path = typer.Argument(..., help="Path to tree JSON file", exists=True),
):
    """
    Print tree structure in a readable format.
    
    Example:
        python -m tree print results/biology_structure.json
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        structure = data.get('structure', [])
        doc_name = data.get('doc_name', json_file.stem)
        
        typer.secho(f"\n{doc_name}", fg=typer.colors.CYAN, bold=True)
        typer.echo("=" * 60)
        print_tree(structure)
        typer.echo("=" * 60 + "\n")
        
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

