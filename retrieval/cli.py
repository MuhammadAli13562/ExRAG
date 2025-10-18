"""
CLI interface for the retrieval and agent pipeline.
"""
from pathlib import Path
from typing import Optional
import typer
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import (
    query_agent,
    evaluate_index,
    load_queries_from_file,
    list_collections,
)
from common.models import RetrievalConfig
from common.settings import get_settings
from common.logging_config import get_logger


logger = get_logger("exrag.retrieval.cli")

app = typer.Typer(
    help="Query and evaluate the agentic retrieval system",
    add_completion=False,
)


@app.command("query")
def cli_query(
    query: str = typer.Argument(..., help="Query string"),
    title_collection: str = typer.Option(..., "--title", "-t", help="Title-indexed collection name"),
    text_collection: str = typer.Option(..., "--text", "-x", help="Text-indexed collection name"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of results to retrieve"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iter", help="Max agent iterations"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output with intermediate steps"),
):
    """
    Run a single query through the agent.
    
    Examples:
        python -m retrieval query "What are electric charges?" \\
            --title biology_titles --text biology_texts
        
        python -m retrieval query "Explain photosynthesis" \\
            --title bio_titles --text bio_texts --top-k 10 --verbose
    """
    try:
        # Build config
        settings = get_settings()
        config = RetrievalConfig(
            agent_model=settings.agent_model,
            temperature=settings.agent_temperature,
            top_k=top_k or settings.retrieval_top_k,
            similarity_threshold=settings.retrieval_similarity_threshold,
            max_iterations=max_iterations or settings.max_agent_iterations,
            recursion_limit=settings.agent_recursion_limit,
            exploration_count=settings.default_exploration_count,
        )
        
        result = query_agent(
            query=query,
            title_collection=title_collection,
            text_collection=text_collection,
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
            typer.secho("="*80, fg=typer.colors.CYAN)
            typer.echo(f"\nIterations: {result.iteration_count}")
            typer.echo(f"Title collection: {result.title_collection}")
            typer.echo(f"Text collection: {result.text_collection}\n")
        
        raise typer.Exit(0)
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("interactive")
def cli_interactive(
    title_collection: str = typer.Option(..., "--title", "-t", help="Title-indexed collection name"),
    text_collection: str = typer.Option(..., "--text", "-x", help="Text-indexed collection name"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of results to retrieve"),
):
    """
    Run the agent in interactive mode.
    
    Example:
        python -m retrieval interactive --title bio_titles --text bio_texts
    """
    typer.secho("\n" + "="*80, fg=typer.colors.CYAN, bold=True)
    typer.secho("AGENTIC RETRIEVAL SYSTEM - Interactive Mode", fg=typer.colors.CYAN, bold=True)
    typer.secho("="*80, fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Title Collection: {title_collection}")
    typer.echo(f"Text Collection: {text_collection}")
    typer.echo("\nType 'exit' or 'quit' to end the session")
    typer.secho("="*80 + "\n", fg=typer.colors.CYAN, bold=True)
    
    settings = get_settings()
    config = RetrievalConfig(
        agent_model=settings.agent_model,
        temperature=settings.agent_temperature,
        top_k=top_k or settings.retrieval_top_k,
        similarity_threshold=settings.retrieval_similarity_threshold,
        max_iterations=settings.max_agent_iterations,
        recursion_limit=settings.agent_recursion_limit,
        exploration_count=settings.default_exploration_count,
    )
    
    while True:
        try:
            user_input = typer.prompt("\n🔍 Your query", default="").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                typer.echo("\nGoodbye! 👋\n")
                break
            
            # Query the agent
            result = query_agent(
                query=user_input,
                title_collection=title_collection,
                text_collection=text_collection,
                config=config,
                verbose=True,
            )
            
            if not result.success:
                typer.secho(f"\n✗ Error: {result.error}\n", fg=typer.colors.RED)
            
        except KeyboardInterrupt:
            typer.echo("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            typer.secho(f"\n✗ Error: {e}\n", fg=typer.colors.RED)


@app.command("eval")
def cli_eval(
    queries_file: Path = typer.Argument(..., help="Path to queries file (one per line or JSONL)", exists=True),
    title_collection: str = typer.Option(..., "--title", "-t", help="Title-indexed collection name"),
    text_collection: str = typer.Option(..., "--text", "-x", help="Text-indexed collection name"),
    chroma_dir: Optional[Path] = typer.Option(None, "--db", help="ChromaDB directory (default from settings)"),
    max_queries: Optional[int] = typer.Option(None, "--max", "-n", help="Max queries to evaluate"),
    save_results: Optional[Path] = typer.Option(None, "--save", "-s", help="Save detailed results to JSONL"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of results to retrieve"),
    json_output: bool = typer.Option(False, "--json", help="Output metrics as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """
    Evaluate retrieval quality on a set of queries.
    
    Examples:
        python -m retrieval eval queries.txt --title bio_titles --text bio_texts
        
        python -m retrieval eval queries.jsonl --title bio_titles --text bio_texts \\
            --max 10 --save results/eval_results.jsonl
    """
    if verbose:
        logger.setLevel("DEBUG")
    
    try:
        # Load queries
        queries = load_queries_from_file(queries_file)
        
        if not queries:
            typer.secho("✗ No queries found in file", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        
        # Build config
        settings = get_settings()
        config = RetrievalConfig(
            agent_model=settings.agent_model,
            temperature=settings.agent_temperature,
            top_k=top_k or settings.retrieval_top_k,
            similarity_threshold=settings.retrieval_similarity_threshold,
            max_iterations=settings.max_agent_iterations,
            recursion_limit=settings.agent_recursion_limit,
            exploration_count=settings.default_exploration_count,
        )
        
        typer.secho(f"\nEvaluating {len(queries)} queries...", fg=typer.colors.CYAN)
        
        metrics = evaluate_index(
            queries=queries,
            title_collection=title_collection,
            text_collection=text_collection,
            chroma_dir=chroma_dir,
            config=config,
            max_queries=max_queries,
            save_results=save_results,
        )
        
        if json_output:
            print(metrics.model_dump_json(indent=2))
        else:
            typer.secho("\n✓ Evaluation complete", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"\n{'='*60}")
            typer.echo("Evaluation Metrics")
            typer.echo(f"{'='*60}")
            typer.echo(f"Total queries: {metrics.total_queries}")
            typer.echo(f"Successful: {metrics.successful}")
            typer.echo(f"Failed: {metrics.failed}")
            typer.echo(f"Success rate: {metrics.success_rate:.1%}")
            typer.echo(f"Avg iterations: {metrics.avg_iteration_count:.1f}")
            typer.echo(f"Avg response length: {metrics.avg_response_length:.0f} chars")
            
            if save_results:
                typer.echo(f"\nDetailed results: {save_results}")
            
            typer.echo(f"{'='*60}\n")
        
        raise typer.Exit(0)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=verbose)
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command("list")
def cli_list(
    chroma_dir: Optional[Path] = typer.Option(None, "--db", help="ChromaDB directory (default from settings)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List all available collections.
    
    Example:
        python -m retrieval list
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
            
            # Group by index field
            title_collections = [c for c in collections if c['metadata'].get('index_field') == 'title']
            text_collections = [c for c in collections if c['metadata'].get('index_field') == 'text']
            
            if title_collections:
                typer.secho("📚 Title-Indexed Collections:", fg=typer.colors.GREEN, bold=True)
                for coll in title_collections:
                    typer.echo(f"  • {coll['name']} ({coll['count']} vectors)")
                    if 'json_source' in coll['metadata']:
                        typer.echo(f"    Source: {coll['metadata']['json_source']}")
                typer.echo()
            
            if text_collections:
                typer.secho("📄 Text-Indexed Collections:", fg=typer.colors.GREEN, bold=True)
                for coll in text_collections:
                    typer.echo(f"  • {coll['name']} ({coll['count']} vectors)")
                    if 'json_source' in coll['metadata']:
                        typer.echo(f"    Source: {coll['metadata']['json_source']}")
                typer.echo()
            
            typer.echo(f"{'='*80}\n")
        
        raise typer.Exit(0)
        
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

