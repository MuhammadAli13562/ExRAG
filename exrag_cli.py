"""
Top-level CLI wrapper for the ExRAG pipeline.
Provides a unified interface to all pipeline stages.
"""
from pathlib import Path
from typing import Optional
import typer
import sys

sys.path.insert(0, str(Path(__file__).parent))

from common.settings import get_settings
from common.logging_config import get_logger

logger = get_logger("exrag.cli")

app = typer.Typer(
    help="ExRAG: Hierarchical document indexing and agentic retrieval pipeline",
    add_completion=False,
)


# ===== Tree Building Commands =====

@app.command("tree")
def tree_command(
    md_file: Path = typer.Argument(..., help="Path to markdown file", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path"),
    mode: str = typer.Option("full", "--mode", "-m", help="Build mode (simple or full)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """
    Build tree structure from markdown file.
    
    Example:
        exrag tree md/biology.md
        exrag tree md/physics.md --output results/custom.json
    """
    from tree.pipeline import build_tree
    
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
            status = "⊙ Skipped (unchanged)" if result.skipped else "✓ Built successfully"
            color = typer.colors.YELLOW if result.skipped else typer.colors.GREEN
            typer.secho(f"{status}: {result.output_path}", fg=color)
            typer.echo(f"  Total nodes: {result.total_nodes}, Max depth: {result.max_depth}")
        
    except Exception as e:
        logger.error(f"Tree build failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ===== Embedding Commands =====

@app.command("embed")
def embed_command(
    tree_json: Path = typer.Argument(..., help="Path to tree JSON file", exists=True),
    which: str = typer.Option("both", "--which", "-w", help="Which indexes to build (titles, texts, or both)"),
    prefix: Optional[str] = typer.Option(None, "--prefix", "-p", help="Collection name prefix"),
    reset: bool = typer.Option(False, "--reset", "-r", help="Reset existing collections"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """
    Build embedding indexes from tree JSON.
    
    Example:
        exrag embed results/biology_structure.json
        exrag embed results/physics_structure.json --which titles --reset
    """
    from embeddings.pipeline import build_embedding_indexes
    from common.models import EmbeddingConfig
    
    if verbose:
        logger.setLevel("DEBUG")
    
    try:
        settings = get_settings()
        config = EmbeddingConfig(
            model=model or settings.embedding_model,
            provider=settings.embedding_provider,
            batch_size=settings.embedding_batch_size,
        )
        
        result = build_embedding_indexes(
            tree_json=tree_json,
            config=config,
            which=which,
            collection_prefix=prefix,
            reset=reset,
        )
        
        if json_output:
            print(result.model_dump_json(indent=2))
        else:
            typer.secho("✓ Embedding indexes built", fg=typer.colors.GREEN)
            typer.echo(f"  Total indexed: {result.total_indexed} vectors")
            if result.title_stats:
                typer.echo(f"  Title collection: {result.title_stats.collection_name}")
            if result.text_stats:
                typer.echo(f"  Text collection: {result.text_stats.collection_name}")
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ===== Query Commands =====

@app.command("query")
def query_command(
    query: Optional[str] = typer.Argument(None, help="Query string (omit for --loop mode)"),
    title: str = typer.Option(..., "--title", "-t", help="Title collection name"),
    text: str = typer.Option(..., "--text", "-x", help="Text collection name"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of results"),
    loop: bool = typer.Option(False, "--loop", "-l", help="Interactive loop mode"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Query the retrieval agent.
    
    Examples:
        exrag query "What are electric charges?" --title bio_titles --text bio_texts
        exrag query --loop --title bio_titles --text bio_texts
    """
    from retrieval.pipeline import query_agent
    from common.models import RetrievalConfig
    
    settings = get_settings()
    config = RetrievalConfig(
        top_k=top_k or settings.retrieval_top_k,
    )
    
    # Interactive loop mode
    if loop:
        typer.secho("\n" + "="*80, fg=typer.colors.GREEN, bold=True)
        typer.secho("🔍 INTERACTIVE QUERY MODE", fg=typer.colors.GREEN, bold=True)
        typer.secho("="*80, fg=typer.colors.GREEN, bold=True)
        typer.echo(f"📚 Title Collection: {title}")
        typer.echo(f"📄 Text Collection: {text}")
        typer.echo(f"🔢 Top-K: {config.top_k}")
        typer.secho("\nType your questions below. Type 'exit' or 'quit' to stop.", fg=typer.colors.YELLOW)
        typer.secho("="*80 + "\n", fg=typer.colors.GREEN, bold=True)
        
        question_count = 0
        
        while True:
            try:
                # Get query from user
                user_query = typer.prompt("\n🔎 Your question", default="").strip()
                
                if not user_query:
                    typer.secho("⚠️  Please enter a question.", fg=typer.colors.YELLOW)
                    continue
                
                # Check for exit commands
                if user_query.lower() in ['exit', 'quit', 'q', 'bye']:
                    typer.secho(f"\n👋 Goodbye! Asked {question_count} question(s).", fg=typer.colors.GREEN)
                    break
                
                question_count += 1
                
                # Execute query
                try:
                    result = query_agent(
                        query=user_query,
                        title_collection=title,
                        text_collection=text,
                        config=config,
                        verbose=verbose,
                    )
                    
                    if json_output:
                        print(result.model_dump_json(indent=2))
                    else:
                        if not result.success:
                            typer.secho(f"\n✗ Query failed: {result.error}", fg=typer.colors.RED, err=True)
                            continue
                        
                        typer.secho("\n" + "="*80, fg=typer.colors.CYAN)
                        typer.secho("💡 ANSWER", fg=typer.colors.CYAN, bold=True)
                        typer.secho("="*80, fg=typer.colors.CYAN)
                        typer.echo(result.answer)
                        typer.secho("="*80, fg=typer.colors.CYAN)
                
                except Exception as e:
                    logger.error(f"Query failed: {e}", exc_info=verbose)
                    typer.secho(f"\n✗ Error: {e}", fg=typer.colors.RED, err=True)
                    typer.secho("💡 Try another question or type 'exit' to quit.", fg=typer.colors.YELLOW)
            
            except KeyboardInterrupt:
                typer.secho(f"\n\n👋 Interrupted! Asked {question_count} question(s).", fg=typer.colors.GREEN)
                break
            except EOFError:
                typer.secho(f"\n\n👋 Session ended. Asked {question_count} question(s).", fg=typer.colors.GREEN)
                break
        
        return
    
    # Single query mode
    if not query:
        typer.secho("✗ Error: Query is required in non-loop mode", fg=typer.colors.RED, err=True)
        typer.secho("💡 Use --loop for interactive mode, or provide a query string", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    
    try:
        result = query_agent(
            query=query,
            title_collection=title,
            text_collection=text,
            config=config,
            verbose=verbose,
        )
        
        if json_output:
            print(result.model_dump_json(indent=2))
        else:
            if not result.success:
                typer.secho(f"✗ Query failed: {result.error}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
            
            typer.secho("\n" + "="*80, fg=typer.colors.CYAN)
            typer.secho("ANSWER", fg=typer.colors.CYAN, bold=True)
            typer.secho("="*80, fg=typer.colors.CYAN)
            typer.echo(result.answer)
            typer.secho("="*80 + "\n", fg=typer.colors.CYAN)
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ===== Pipeline Command (end-to-end) =====

@app.command("pipeline")
def pipeline_command(
    md_file: Path = typer.Argument(..., help="Path to markdown file", exists=True),
    mode: str = typer.Option("full", "--mode", help="Tree build mode (simple or full)"),
    which: str = typer.Option("both", "--which", "-w", help="Which indexes (titles, texts, or both)"),
    prefix: Optional[str] = typer.Option(None, "--prefix", "-p", help="Collection prefix"),
    force_tree: bool = typer.Option(False, "--force-tree", help="Force rebuild tree"),
    reset_indexes: bool = typer.Option(False, "--reset-indexes", help="Reset indexes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """
    Run the full pipeline: tree → embeddings.
    
    Example:
        exrag pipeline md/biology.md
        exrag pipeline md/physics.md --reset-indexes --verbose
    """
    from tree.pipeline import build_tree
    from embeddings.pipeline import build_embedding_indexes
    from common.models import EmbeddingConfig
    
    if verbose:
        logger.setLevel("DEBUG")
    
    try:
        settings = get_settings()
        
        # Step 1: Build tree
        typer.secho("\n[1/2] Building tree...", fg=typer.colors.CYAN, bold=True)
        tree_result = build_tree(
            md_path=md_file,
            mode=mode,
            force=force_tree,
        )
        
        status = "Skipped (unchanged)" if tree_result.skipped else "Built"
        typer.secho(f"  ✓ {status}: {tree_result.output_path}", fg=typer.colors.GREEN)
        
        # Step 2: Build embeddings
        typer.secho("\n[2/2] Building embeddings...", fg=typer.colors.CYAN, bold=True)
        config = EmbeddingConfig(
            model=settings.embedding_model,
            provider=settings.embedding_provider,
            batch_size=settings.embedding_batch_size,
        )
        
        embed_result = build_embedding_indexes(
            tree_json=tree_result.output_path,
            config=config,
            which=which,
            collection_prefix=prefix,
            reset=reset_indexes,
        )
        
        typer.secho(f"  ✓ Indexed {embed_result.total_indexed} vectors", fg=typer.colors.GREEN)
        
        # Summary
        typer.secho("\n" + "="*60, fg=typer.colors.GREEN, bold=True)
        typer.secho("PIPELINE COMPLETE", fg=typer.colors.GREEN, bold=True)
        typer.secho("="*60, fg=typer.colors.GREEN, bold=True)
        typer.echo(f"Document: {tree_result.doc_name}")
        typer.echo(f"Tree: {tree_result.output_path}")
        typer.echo(f"Total nodes: {tree_result.total_nodes}")
        typer.echo(f"Vectors indexed: {embed_result.total_indexed}")
        
        if embed_result.title_stats:
            typer.echo(f"\nTitle collection: {embed_result.title_stats.collection_name}")
        if embed_result.text_stats:
            typer.echo(f"Text collection: {embed_result.text_stats.collection_name}")
        
        typer.secho("\n" + "="*60 + "\n", fg=typer.colors.GREEN, bold=True)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=verbose)
        typer.secho(f"\n✗ Pipeline failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ===== List Collections =====

@app.command("list")
def list_command(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List all collections in ChromaDB.
    
    Example:
        exrag list
    """
    from embeddings.pipeline import list_collections
    
    try:
        collections = list_collections()
        
        if json_output:
            import json
            print(json.dumps(collections, indent=2, default=str))
        else:
            if not collections:
                typer.echo("No collections found.")
                return
            
            typer.secho(f"\nFound {len(collections)} collection(s):\n", fg=typer.colors.CYAN, bold=True)
            
            for coll in collections:
                name = coll.get('name', 'Unknown')
                count = coll.get('count', 0)
                index_field = coll.get('metadata', {}).get('index_field', 'unknown')
                
                typer.secho(f"  • {name}", fg=typer.colors.GREEN, bold=True)
                typer.echo(f"    Vectors: {count}, Field: {index_field}")
            
            typer.echo()
        
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ===== Dashboard Command =====

@app.command("dashboard")
def dashboard_command(
    port: int = typer.Option(8501, "--port", "-p", help="Dashboard port"),
):
    """
    Launch the interactive query dashboard.
    
    Example:
        exrag dashboard
        exrag dashboard --port 8502
    """
    import subprocess
    
    try:
        typer.secho("🚀 Starting Vector Query Dashboard...", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"📊 Dashboard will be available at: http://localhost:{port}")
        typer.echo()
        
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "embeddings/dashboard/app.py",
            "--server.port",
            str(port),
            "--server.address",
            "localhost"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        typer.echo("\n👋 Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        typer.secho(f"✗ Error running dashboard: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        typer.secho("💡 Tip: Make sure streamlit is installed: pip install streamlit", fg=typer.colors.YELLOW)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

