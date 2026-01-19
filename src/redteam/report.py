"""Report generation for red team evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from redteam.evaluator import EvaluationResult


@dataclass
class VulnerabilityResult:
    """Results for a single vulnerability type."""

    name: str
    attacks: int
    passed: int
    failed: int

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate for this vulnerability."""
        if self.attacks == 0:
            return 0.0
        return self.passed / self.attacks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "attacks": self.attacks,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
        }


@dataclass
class ModelReport:
    """Report for a single model evaluation."""

    model_name: str
    timestamp: str
    risk_score: float
    total_attacks: int
    total_passed: int
    total_failed: int
    pass_rate: float
    vulnerability_results: list[VulnerabilityResult] = field(default_factory=list)
    failed_examples: list[dict] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: EvaluationResult) -> ModelReport:
        """Create a ModelReport from an EvaluationResult."""
        return cls(
            model_name=result.model_name,
            timestamp=result.timestamp.isoformat(),
            risk_score=result.risk_score,
            total_attacks=result.total_attacks,
            total_passed=result.total_passed,
            total_failed=result.total_failed,
            pass_rate=result.pass_rate,
            vulnerability_results=result.vulnerability_results,
            failed_examples=result.failed_examples,
            config={
                "preset": result.config.preset.value,
                "attacks_per_vulnerability": result.config.attacks_per_vulnerability,
                "enable_multi_turn": result.config.enable_multi_turn,
                "purpose": result.config.purpose,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "risk_score": self.risk_score,
            "summary": {
                "total_attacks": self.total_attacks,
                "total_passed": self.total_passed,
                "total_failed": self.total_failed,
                "pass_rate": self.pass_rate,
            },
            "vulnerability_results": [vr.to_dict() for vr in self.vulnerability_results],
            "failed_examples": self.failed_examples[:10],  # Limit for readability
            "config": self.config,
        }


@dataclass
class ComparisonReport:
    """Report comparing base and trained model evaluations."""

    base_report: ModelReport
    trained_report: ModelReport
    improvements: dict[str, float] = field(default_factory=dict)
    overall_improvement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_model": self.base_report.to_dict(),
            "trained_model": self.trained_report.to_dict(),
            "comparison": {
                "overall_improvement": self.overall_improvement,
                "improvements_by_vulnerability": self.improvements,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


class ReportPrinter:
    """Prints formatted reports to the console."""

    def __init__(self):
        """Initialize the report printer."""
        self.console = Console()

    def _risk_color(self, risk_score: float) -> str:
        """Get color for risk score."""
        if risk_score < 0.2:
            return "green"
        if risk_score < 0.4:
            return "yellow"
        return "red"

    def _risk_label(self, risk_score: float) -> str:
        """Get label for risk score."""
        if risk_score < 0.2:
            return "Low"
        if risk_score < 0.4:
            return "Medium"
        return "High"

    def _improvement_color(self, improvement: float) -> str:
        """Get color for improvement value."""
        if improvement > 0.1:
            return "green"
        if improvement > 0:
            return "yellow"
        return "red"

    def print_model_report(self, report: ModelReport) -> None:
        """Print a single model report."""
        # Header
        self.console.print()
        self.console.rule("[bold]DEEPTEAM RED TEAM EVALUATION REPORT[/bold]", style="blue")
        self.console.print()

        # Model info
        self.console.print(f"[bold]Model:[/bold] {report.model_name}")
        self.console.print(f"[bold]Date:[/bold] {report.timestamp}")
        self.console.print(f"[bold]Purpose:[/bold] {report.config.get('purpose', 'N/A')}")
        self.console.print()

        # Risk score
        risk_color = self._risk_color(report.risk_score)
        risk_label = self._risk_label(report.risk_score)
        self.console.print(
            Panel(
                Text(f"{report.risk_score:.2f} ({risk_label})", style=f"bold {risk_color}"),
                title="Overall Risk Score",
                border_style=risk_color,
            )
        )
        self.console.print()

        # Vulnerability table
        self.console.print("[bold]VULNERABILITY RESULTS[/bold]")
        self.console.rule(style="dim")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Vulnerability", style="cyan")
        table.add_column("Attacks", justify="right")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Pass Rate", justify="right")

        for vr in report.vulnerability_results:
            pass_rate_color = "green" if vr.pass_rate >= 0.8 else "yellow" if vr.pass_rate >= 0.5 else "red"
            table.add_row(
                vr.name,
                str(vr.attacks),
                str(vr.passed),
                str(vr.failed),
                f"[{pass_rate_color}]{vr.pass_rate:.0%}[/{pass_rate_color}]",
            )

        # Summary row
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{report.total_attacks}[/bold]",
            f"[bold]{report.total_passed}[/bold]",
            f"[bold]{report.total_failed}[/bold]",
            f"[bold]{report.pass_rate:.0%}[/bold]",
            style="dim",
        )

        self.console.print(table)
        self.console.print()

        # Failed examples
        if report.failed_examples:
            self.console.print("[bold]FAILED EXAMPLES[/bold]")
            self.console.rule(style="dim")

            for i, example in enumerate(report.failed_examples[:5], 1):
                self.console.print(f"\n[bold red][{i}] {example['vulnerability']} via {example['attack_type']}[/bold red]")

                input_text = example.get("input", "")[:300]
                if len(example.get("input", "")) > 300:
                    input_text += "..."
                self.console.print(f"  [dim]Input:[/dim] {input_text}")

                output_text = example.get("output", "")[:300]
                if len(example.get("output", "")) > 300:
                    output_text += "..."
                self.console.print(f"  [dim]Output:[/dim] {output_text}")

        self.console.print()
        self.console.rule(style="blue")

    def print_comparison_report(self, report: ComparisonReport) -> None:
        """Print a comparison report between base and trained models."""
        # Header
        self.console.print()
        self.console.rule("[bold]DEEPTEAM RED TEAM COMPARISON REPORT[/bold]", style="blue")
        self.console.print()

        # Model info
        self.console.print(f"[bold]Base Model:[/bold] {report.base_report.model_name}")
        self.console.print(f"[bold]Trained Model:[/bold] {report.trained_report.model_name}")
        self.console.print(f"[bold]Date:[/bold] {report.trained_report.timestamp}")
        self.console.print()

        # Overall comparison panel
        improvement_color = self._improvement_color(report.overall_improvement)
        improvement_sign = "+" if report.overall_improvement > 0 else ""

        base_color = self._risk_color(report.base_report.risk_score)
        trained_color = self._risk_color(report.trained_report.risk_score)

        comparison_text = Text()
        comparison_text.append("Base Risk: ", style="dim")
        comparison_text.append(f"{report.base_report.risk_score:.2f}", style=f"bold {base_color}")
        comparison_text.append("  ->  ", style="dim")
        comparison_text.append("Trained Risk: ", style="dim")
        comparison_text.append(f"{report.trained_report.risk_score:.2f}", style=f"bold {trained_color}")
        comparison_text.append(f"\n\nImprovement: ", style="dim")
        comparison_text.append(
            f"{improvement_sign}{report.overall_improvement:.1%}",
            style=f"bold {improvement_color}",
        )

        self.console.print(
            Panel(
                comparison_text,
                title="Overall Comparison",
                border_style=improvement_color,
            )
        )
        self.console.print()

        # Vulnerability comparison table
        self.console.print("[bold]COMPARISON BY VULNERABILITY[/bold]")
        self.console.rule(style="dim")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Vulnerability", style="cyan")
        table.add_column("Base Pass Rate", justify="right")
        table.add_column("Trained Pass Rate", justify="right")
        table.add_column("Improvement", justify="right")

        for base_vr in report.base_report.vulnerability_results:
            trained_vr = None
            for tvr in report.trained_report.vulnerability_results:
                if tvr.name == base_vr.name:
                    trained_vr = tvr
                    break

            if trained_vr:
                improvement = report.improvements.get(base_vr.name, 0)
                imp_color = self._improvement_color(improvement)
                imp_sign = "+" if improvement > 0 else ""

                base_color = "green" if base_vr.pass_rate >= 0.8 else "yellow" if base_vr.pass_rate >= 0.5 else "red"
                trained_color = "green" if trained_vr.pass_rate >= 0.8 else "yellow" if trained_vr.pass_rate >= 0.5 else "red"

                table.add_row(
                    base_vr.name,
                    f"[{base_color}]{base_vr.pass_rate:.0%}[/{base_color}]",
                    f"[{trained_color}]{trained_vr.pass_rate:.0%}[/{trained_color}]",
                    f"[{imp_color}]{imp_sign}{improvement:.0%}[/{imp_color}]",
                )

        # Summary row
        base_rate = report.base_report.pass_rate
        trained_rate = report.trained_report.pass_rate
        overall_imp = report.overall_improvement
        imp_color = self._improvement_color(overall_imp)
        imp_sign = "+" if overall_imp > 0 else ""

        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{base_rate:.0%}[/bold]",
            f"[bold]{trained_rate:.0%}[/bold]",
            f"[bold {imp_color}]{imp_sign}{overall_imp:.0%}[/bold {imp_color}]",
            style="dim",
        )

        self.console.print(table)
        self.console.print()

        # Failed examples from trained model (areas still needing work)
        if report.trained_report.failed_examples:
            self.console.print("[bold]REMAINING VULNERABILITIES (Trained Model Failures)[/bold]")
            self.console.rule(style="dim")

            for i, example in enumerate(report.trained_report.failed_examples[:3], 1):
                self.console.print(f"\n[bold red][{i}] {example['vulnerability']} via {example['attack_type']}[/bold red]")

                input_text = example.get("input", "")[:200]
                if len(example.get("input", "")) > 200:
                    input_text += "..."
                self.console.print(f"  [dim]Input:[/dim] {input_text}")

                output_text = example.get("output", "")[:200]
                if len(example.get("output", "")) > 200:
                    output_text += "..."
                self.console.print(f"  [dim]Output:[/dim] {output_text}")

        self.console.print()
        self.console.rule(style="blue")
