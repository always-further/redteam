"""DeepTeam evaluation runner with base model comparison."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from deepteam import red_team
from deepteam.attacks.multi_turn import (
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
)
from deepteam.attacks.single_turn import Leetspeak, MathProblem, PromptInjection, ROT13
from deepteam.vulnerabilities import (
    Bias,
    Competition,
    ExcessiveAgency,
    GraphicContent,
    IllegalActivity,
    Misinformation,
    PersonalSafety,
    PIILeakage,
    PromptLeakage,
    ShellInjection,
    SQLInjection,
    SSRF,
)

from redteam.model_client import VLLMClient, create_model_callback
from redteam.report import ComparisonReport, ModelReport, VulnerabilityResult


class VulnerabilityPreset(Enum):
    """Predefined vulnerability sets for different evaluation scenarios."""

    # Security-focused (matches Hedgehog attack types)
    SECURITY = "security"

    # Full red team evaluation
    FULL = "full"

    # Quick smoke test
    QUICK = "quick"


@dataclass
class EvaluationConfig:
    """Configuration for red team evaluation."""

    # Attack configuration
    attacks_per_vulnerability: int = 5
    max_concurrent: int = 10

    # Model generation settings
    max_tokens: int = 512
    temperature: float = 0.7

    # Vulnerability preset
    preset: VulnerabilityPreset = VulnerabilityPreset.SECURITY

    # Purpose description for attack generation
    purpose: str = "A coding assistant with file system access"

    # Enable multi-turn attacks (slower but more thorough)
    enable_multi_turn: bool = True


@dataclass
class EvaluationResult:
    """Results from a single model evaluation."""

    model_name: str
    timestamp: datetime
    config: EvaluationConfig
    vulnerability_results: list[VulnerabilityResult] = field(default_factory=list)
    total_attacks: int = 0
    total_passed: int = 0
    total_failed: int = 0
    failed_examples: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate overall pass rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.total_passed / self.total_attacks

    @property
    def risk_score(self) -> float:
        """Calculate risk score (0 = safe, 1 = vulnerable)."""
        return 1.0 - self.pass_rate


def get_vulnerabilities(preset: VulnerabilityPreset):
    """Get vulnerability instances for a preset.

    Args:
        preset: The vulnerability preset to use

    Returns:
        List of vulnerability instances
    """
    if preset == VulnerabilityPreset.QUICK:
        return [
            ShellInjection(),
            PromptInjection(),
        ]

    if preset == VulnerabilityPreset.SECURITY:
        return [
            # Injection attacks (maps to Hedgehog INDIRECT_INJECTION, MASS_DELETION)
            ShellInjection(),
            SQLInjection(),
            PromptInjection(),
            # Data exfiltration (maps to EXTERNAL_EXFILTRATION, CREDENTIAL_ACCESS)
            PIILeakage(types=["api_key", "password", "ssh_key", "credit_card"]),
            PromptLeakage(),
            SSRF(),
            # Agentic risks (maps to MULTI_STEP_ATTACK)
            ExcessiveAgency(),
        ]

    # FULL preset
    return [
        # Security
        ShellInjection(),
        SQLInjection(),
        PromptInjection(),
        PIILeakage(types=["api_key", "password", "ssh_key", "credit_card", "email", "phone"]),
        PromptLeakage(),
        SSRF(),
        ExcessiveAgency(),
        # Safety
        IllegalActivity(),
        GraphicContent(),
        PersonalSafety(),
        # Responsible AI
        Bias(types=["race", "gender", "religion", "political"]),
        Misinformation(),
        Competition(),
    ]


def get_attacks(config: EvaluationConfig):
    """Get attack instances based on configuration.

    Args:
        config: Evaluation configuration

    Returns:
        List of attack instances
    """
    attacks = [
        # Single-turn attacks
        PromptInjection(),
        ROT13(),
        Leetspeak(),
        MathProblem(),
    ]

    if config.enable_multi_turn:
        attacks.extend([
            LinearJailbreaking(),
            TreeJailbreaking(),
            CrescendoJailbreaking(),
        ])

    return attacks


class RedTeamEvaluator:
    """Evaluates models using DeepTeam red teaming."""

    def __init__(self, config: EvaluationConfig | None = None):
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()

    async def evaluate_model(
        self,
        client: VLLMClient,
        model_name: str,
    ) -> EvaluationResult:
        """Evaluate a single model.

        Args:
            client: VLLMClient connected to the model
            model_name: Name/path of the model for reporting

        Returns:
            EvaluationResult with all metrics
        """
        callback = create_model_callback(
            client,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        vulnerabilities = get_vulnerabilities(self.config.preset)
        attacks = get_attacks(self.config)

        # Run DeepTeam evaluation
        risk_assessment = await asyncio.to_thread(
            red_team,
            model_callback=callback,
            vulnerabilities=vulnerabilities,
            attacks=attacks,
            attacks_per_vulnerability=self.config.attacks_per_vulnerability,
            max_concurrent=self.config.max_concurrent,
            purpose=self.config.purpose,
        )

        # Parse results
        result = EvaluationResult(
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            config=self.config,
        )

        for vuln_result in risk_assessment.vulnerability_scores:
            vuln_name = vuln_result.vulnerability.__class__.__name__
            passed = vuln_result.score == 1  # DeepTeam uses 1 for pass, 0 for fail
            attacks_count = len(vuln_result.attack_results) if vuln_result.attack_results else 1

            vr = VulnerabilityResult(
                name=vuln_name,
                attacks=attacks_count,
                passed=attacks_count if passed else 0,
                failed=0 if passed else attacks_count,
            )
            result.vulnerability_results.append(vr)
            result.total_attacks += attacks_count
            result.total_passed += vr.passed
            result.total_failed += vr.failed

            # Collect failed examples
            if not passed and vuln_result.attack_results:
                for attack_result in vuln_result.attack_results[:3]:  # Limit to 3 per vuln
                    if attack_result.score == 0:
                        result.failed_examples.append({
                            "vulnerability": vuln_name,
                            "attack_type": attack_result.attack.__class__.__name__,
                            "input": attack_result.input[:500] if attack_result.input else "",
                            "output": attack_result.output[:500] if attack_result.output else "",
                        })

        return result

    async def evaluate_comparison(
        self,
        base_client: VLLMClient,
        trained_client: VLLMClient,
        base_model_name: str,
        trained_model_name: str,
    ) -> ComparisonReport:
        """Evaluate and compare base and trained models.

        Args:
            base_client: VLLMClient for base model
            trained_client: VLLMClient for trained model
            base_model_name: Name of base model
            trained_model_name: Name of trained model

        Returns:
            ComparisonReport with both results and improvements
        """
        # Evaluate both models
        base_result = await self.evaluate_model(base_client, base_model_name)
        trained_result = await self.evaluate_model(trained_client, trained_model_name)

        # Calculate improvements
        improvements = {}
        for base_vr in base_result.vulnerability_results:
            for trained_vr in trained_result.vulnerability_results:
                if base_vr.name == trained_vr.name:
                    base_rate = base_vr.pass_rate
                    trained_rate = trained_vr.pass_rate
                    improvements[base_vr.name] = trained_rate - base_rate

        overall_improvement = trained_result.pass_rate - base_result.pass_rate

        return ComparisonReport(
            base_report=ModelReport.from_result(base_result),
            trained_report=ModelReport.from_result(trained_result),
            improvements=improvements,
            overall_improvement=overall_improvement,
        )

    def evaluate_model_sync(
        self,
        client: VLLMClient,
        model_name: str,
    ) -> EvaluationResult:
        """Synchronous wrapper for evaluate_model."""
        return asyncio.run(self.evaluate_model(client, model_name))

    def evaluate_comparison_sync(
        self,
        base_client: VLLMClient,
        trained_client: VLLMClient,
        base_model_name: str,
        trained_model_name: str,
    ) -> ComparisonReport:
        """Synchronous wrapper for evaluate_comparison."""
        return asyncio.run(
            self.evaluate_comparison(
                base_client,
                trained_client,
                base_model_name,
                trained_model_name,
            )
        )
