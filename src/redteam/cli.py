"""CLI for red team evaluation."""

from pathlib import Path
from typing import Annotated

import typer

from redteam.evaluator import EvaluationConfig, RedTeamEvaluator, VulnerabilityPreset
from redteam.model_client import ModelConfig, ModelServer, VLLMClient
from redteam.report import ReportPrinter

app = typer.Typer(
    name="redteam",
    help="Red team evaluation for Hedgehog-trained models using DeepTeam",
    no_args_is_help=True,
)


@app.command()
def evaluate(
    model: Annotated[str, typer.Argument(help="Path to model or HuggingFace model ID")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for JSON report"),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", "-p", help="Vulnerability preset: quick, security, full"),
    ] = "security",
    attacks: Annotated[
        int,
        typer.Option("--attacks", "-a", help="Attacks per vulnerability"),
    ] = 5,
    port: Annotated[
        int,
        typer.Option("--port", help="Port for vLLM server"),
    ] = 8000,
    vllm_url: Annotated[
        str | None,
        typer.Option("--vllm-url", help="URL of existing vLLM server (skip starting one)"),
    ] = None,
    purpose: Annotated[
        str,
        typer.Option("--purpose", help="Purpose description for attack generation"),
    ] = "A coding assistant with file system access",
    no_multi_turn: Annotated[
        bool,
        typer.Option("--no-multi-turn", help="Disable multi-turn attacks (faster)"),
    ] = False,
) -> None:
    """Evaluate a single model with red team attacks."""
    printer = ReportPrinter()

    # Parse preset
    try:
        vuln_preset = VulnerabilityPreset(preset)
    except ValueError:
        printer.console.print(f"[red]Invalid preset: {preset}. Use: quick, security, full[/red]")
        raise typer.Exit(1)

    config = EvaluationConfig(
        preset=vuln_preset,
        attacks_per_vulnerability=attacks,
        purpose=purpose,
        enable_multi_turn=not no_multi_turn,
    )

    evaluator = RedTeamEvaluator(config)

    if vllm_url:
        # Use existing vLLM server
        printer.console.print(f"[dim]Connecting to existing vLLM server at {vllm_url}...[/dim]")
        client = VLLMClient(vllm_url)

        result = evaluator.evaluate_model_sync(client, model)

        from redteam.report import ModelReport

        report = ModelReport.from_result(result)
        printer.print_model_report(report)

        if output:
            import json

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report.to_dict(), indent=2))
            printer.console.print(f"\n[green]Report saved to {output}[/green]")
    else:
        # Start vLLM server
        printer.console.print(f"[dim]Starting vLLM server for {model}...[/dim]")
        model_config = ModelConfig(model_path=model, port=port)

        with ModelServer(model_config) as client:
            printer.console.print("[green]Server ready. Starting evaluation...[/green]\n")

            result = evaluator.evaluate_model_sync(client, model)

            from redteam.report import ModelReport

            report = ModelReport.from_result(result)
            printer.print_model_report(report)

            if output:
                import json

                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report.to_dict(), indent=2))
                printer.console.print(f"\n[green]Report saved to {output}[/green]")


@app.command()
def compare(
    base_model: Annotated[str, typer.Argument(help="Path to base model")],
    trained_model: Annotated[str, typer.Argument(help="Path to Hedgehog-trained model")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for JSON report"),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", "-p", help="Vulnerability preset: quick, security, full"),
    ] = "security",
    attacks: Annotated[
        int,
        typer.Option("--attacks", "-a", help="Attacks per vulnerability"),
    ] = 5,
    base_port: Annotated[
        int,
        typer.Option("--base-port", help="Port for base model vLLM server"),
    ] = 8000,
    trained_port: Annotated[
        int,
        typer.Option("--trained-port", help="Port for trained model vLLM server"),
    ] = 8001,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="URL of existing vLLM server for base model"),
    ] = None,
    trained_url: Annotated[
        str | None,
        typer.Option("--trained-url", help="URL of existing vLLM server for trained model"),
    ] = None,
    purpose: Annotated[
        str,
        typer.Option("--purpose", help="Purpose description for attack generation"),
    ] = "A coding assistant with file system access",
    no_multi_turn: Annotated[
        bool,
        typer.Option("--no-multi-turn", help="Disable multi-turn attacks (faster)"),
    ] = False,
) -> None:
    """Compare base model vs Hedgehog-trained model."""
    printer = ReportPrinter()

    # Parse preset
    try:
        vuln_preset = VulnerabilityPreset(preset)
    except ValueError:
        printer.console.print(f"[red]Invalid preset: {preset}. Use: quick, security, full[/red]")
        raise typer.Exit(1)

    config = EvaluationConfig(
        preset=vuln_preset,
        attacks_per_vulnerability=attacks,
        purpose=purpose,
        enable_multi_turn=not no_multi_turn,
    )

    evaluator = RedTeamEvaluator(config)

    # Determine if we need to start servers
    start_base = base_url is None
    start_trained = trained_url is None

    base_client: VLLMClient | None = None
    trained_client: VLLMClient | None = None
    base_server: ModelServer | None = None
    trained_server: ModelServer | None = None

    try:
        # Set up base model
        if start_base:
            printer.console.print(f"[dim]Starting vLLM server for base model {base_model}...[/dim]")
            base_server = ModelServer(ModelConfig(model_path=base_model, port=base_port))
            base_client = base_server.start()
            printer.console.print("[green]Base model server ready.[/green]")
        else:
            printer.console.print(f"[dim]Connecting to base model at {base_url}...[/dim]")
            base_client = VLLMClient(base_url)

        # Set up trained model
        if start_trained:
            printer.console.print(
                f"[dim]Starting vLLM server for trained model {trained_model}...[/dim]"
            )
            trained_server = ModelServer(ModelConfig(model_path=trained_model, port=trained_port))
            trained_client = trained_server.start()
            printer.console.print("[green]Trained model server ready.[/green]")
        else:
            printer.console.print(f"[dim]Connecting to trained model at {trained_url}...[/dim]")
            trained_client = VLLMClient(trained_url)

        printer.console.print("\n[bold]Starting comparison evaluation...[/bold]\n")

        # Run comparison
        comparison = evaluator.evaluate_comparison_sync(
            base_client,
            trained_client,
            base_model,
            trained_model,
        )

        # Print report
        printer.print_comparison_report(comparison)

        # Save if requested
        if output:
            comparison.save(output)
            printer.console.print(f"\n[green]Report saved to {output}[/green]")

    finally:
        # Clean up servers
        if base_server:
            base_server.stop()
        if trained_server:
            trained_server.stop()


@app.command()
def serve(
    model: Annotated[str, typer.Argument(help="Path to model or HuggingFace model ID")],
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to serve on"),
    ] = 8000,
) -> None:
    """Start a vLLM server for a model (for manual testing)."""
    printer = ReportPrinter()
    printer.console.print(f"[dim]Starting vLLM server for {model} on port {port}...[/dim]")

    model_config = ModelConfig(model_path=model, port=port)
    server = ModelServer(model_config)

    try:
        client = server.start()
        printer.console.print(f"[green]Server ready at http://localhost:{port}[/green]")
        printer.console.print("[dim]Press Ctrl+C to stop[/dim]")

        # Keep running until interrupted
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        printer.console.print("\n[dim]Shutting down...[/dim]")
    finally:
        server.stop()


if __name__ == "__main__":
    app()
