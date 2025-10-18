"""
CLI interface for the embedding pipeline.
"""
from pathlib import Path
from typing import Optional, Literal
import typer
import sys

# Add parent to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.pipeline import (
    build_embedding_indexes,
    list_collections,
    delete_collection,
)
from common.models import EmbeddingConfig
from common.settings import get_settings
from common.logging_config import get_logger


logger = get_logger("exrag.embeddings.cli")

app = typer.Typer(
    help="Build and manage embedding indexes from tree JSON files",
    add_completion=False,
)


@app.command("build")
def cli_build(
    tree_json: Path = typer.Argument(..., help="Path to tree JSON file", exists=True),
    chroma_dir: Optional[Path] = typer.Option(None, "--db", help="ChromaDB directory (default from settings)"),
    which: Literal["titles", "texts", "both"] = typer.Option("both", "--which", "-w", help="Which indexes to build"),
    collection_prefix: Optional[str] = typer.Option(None, "--prefix", "-p", help="Custom collection name prefix"),
    reset: bool = typer.Option(False, "--reset", "-r", help="Delete existing collections before indexing"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model (default from settings)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size for embedding"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """
    Build embedding indexes from a tree JSON file.
    
    This will create ChromaDB collections for title and/or text indexes.
    
    Examples:
        # Build both indexes
        python -m embeddings build results/biology_structure.json
        
        # Build only title index with custom prefix
        python -m embeddings build results/physics_structure.json --which titles --prefix physics_v2
        
        # Force rebuild with custom model
        python -m embeddings build results/biology_structure.json --reset --model text-embedding-3-large
    """
    if verbose:
        logger.setLevel("DEBUG")
    
    try:
        # Build config
        settings = get_settings()
        config = EmbeddingConfig(
            model=model or settings.embedding_model,
            provider=settings.embedding_provider,
            batch_size=batch_size or settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            retry_delay=settings.embedding_retry_delay,
        )
        
        result = build_embedding_indexes(
            tree_json=tree_json,
            chroma_dir=chroma_dir,
            config=config,
            which=which,
            collection_prefix=collection_prefix,
            reset=reset,
        )
        
        if json_output:
            print(result.model_dump_json(indent=2))
        else:
            typer.secho("✓ Embedding indexes built successfully", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"\n{'='*60}")
            typer.echo("Summary")
            typer.echo(f"{'='*60}")
            typer.echo(f"Source: {result.tree_json}")
            typer.echo(f"ChromaDB: {result.chroma_dir}")
            typer.echo(f"Total indexed: {result.total_indexed} vectors")
            
            if result.title_stats:
                typer.echo(f"\nTitle Index:")
                typer.echo(f"  Collection: {result.title_stats.collection_name}")
                typer.echo(f"  Indexed: {result.title_stats.indexed}/{result.title_stats.total_nodes}")
                typer.echo(f"  Success rate: {result.title_stats.success_rate:.1%}")
            
            if result.text_stats:
                typer.echo(f"\nText Index:")
                typer.echo(f"  Collection: {result.text_stats.collection_name}")
                typer.echo(f"  Indexed: {result.text_stats.indexed}/{result.text_stats.total_nodes}")
                typer.echo(f"  Success rate: {result.text_stats.success_rate:.1%}")
            
            typer.echo(f"\n{'='*60}\n")
        
        raise typer.Exit(0)
        
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("list")
def cli_list(
    chroma_dir: Optional[Path] = typer.Option(None, "--db", help="ChromaDB directory (default from settings)"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
):
    """
    List all collections in ChromaDB.
    
    Example:
        python -m embeddings list
    """
    try:
        collections = list_collections(chroma_dir)
        
        if json_output:
            import json
            print(json.dumps(collections, indent=2, default=str))
        else:
            if not collections:
                typer.echo("No collections found.")
                raise typer.Exit(0)
            
            typer.secho(f"\nFound {len(collections)} collection(s):", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"{'='*80}\n")
            
            for coll in collections:
                name = coll.get('name', 'Unknown')
                count = coll.get('count', 0)
                metadata = coll.get('metadata', {})
                index_field = metadata.get('index_field', 'unknown')
                model = metadata.get('embedding_model', 'unknown')
                
                typer.secho(f"  {name}", fg=typer.colors.GREEN, bold=True)
                typer.echo(f"    Vectors: {count}")
                typer.echo(f"    Index field: {index_field}")
                typer.echo(f"    Model: {model}")
                if 'json_source' in metadata:
                    typer.echo(f"    Source: {metadata['json_source']}")
                typer.echo()
            
            typer.echo(f"{'='*80}\n")
        
        raise typer.Exit(0)
        
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("delete")
def cli_delete(
    collection_name: str = typer.Argument(..., help="Name of collection to delete"),
    chroma_dir: Optional[Path] = typer.Option(None, "--db", help="ChromaDB directory (default from settings)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete a collection from ChromaDB.
    
    Example:
        python -m embeddings delete biology_structure__simple_12345678_titles
    """
    try:
        # Get collection info first
        collections = list_collections(chroma_dir)
        coll_info = next((c for c in collections if c['name'] == collection_name), None)
        
        if not coll_info:
            typer.secho(f"✗ Collection '{collection_name}' not found", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        
        # Confirm deletion
        if not yes:
            typer.secho(f"\n⚠️  WARNING: You are about to delete collection: {collection_name}", fg=typer.colors.YELLOW)
            typer.echo(f"   Vectors: {coll_info.get('count', 0)}")
            typer.echo(f"   Metadata: {coll_info.get('metadata', {})}\n")
            
            confirm = typer.confirm("Are you sure you want to delete this collection?")
            if not confirm:
                typer.echo("Deletion cancelled.")
                raise typer.Exit(0)
        
        # Perform deletion
        success = delete_collection(collection_name, chroma_dir)
        
        if success:
            typer.secho(f"✓ Collection '{collection_name}' deleted successfully", fg=typer.colors.GREEN)
            raise typer.Exit(0)
        else:
            typer.secho(f"✗ Failed to delete collection '{collection_name}'", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("info")
def cli_info(
    collection_name: str = typer.Argument(..., help="Name of collection"),
    chroma_dir: Optional[Path] = typer.Option(None, "--db", help="ChromaDB directory (default from settings)"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
):
    """
    Show detailed information about a collection.
    
    Example:
        python -m embeddings info biology_structure__simple_12345678_titles
    """
    try:
        collections = list_collections(chroma_dir)
        coll_info = next((c for c in collections if c['name'] == collection_name), None)
        
        if not coll_info:
            typer.secho(f"✗ Collection '{collection_name}' not found", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        
        if json_output:
            import json
            print(json.dumps(coll_info, indent=2, default=str))
        else:
            typer.secho(f"\nCollection: {collection_name}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"{'='*60}")
            typer.echo(f"Vector count: {coll_info.get('count', 0)}")
            
            metadata = coll_info.get('metadata', {})
            typer.echo(f"\nMetadata:")
            for key, value in metadata.items():
                typer.echo(f"  {key}: {value}")
            typer.echo(f"{'='*60}\n")
        
        raise typer.Exit(0)
        
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

