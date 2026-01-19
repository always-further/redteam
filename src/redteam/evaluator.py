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
            PromptLeakage(),
        ]

    if preset == VulnerabilityPreset.SECURITY:
        return [
            # Injection attacks (maps to Hedgehog INDIRECT_INJECTION, MASS_DELETION)
            ShellInjection(),
            SQLInjection(),
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
            client.base_url,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        vulnerabilities = get_vulnerabilities(self.config.preset)
        attacks = get_attacks(self.config)

        # Run DeepTeam evaluation
        # Use Gemini if GOOGLE_API_KEY is set, otherwise fall back to OpenAI
        import os

        # Determine which model to use for simulation/evaluation
        # Priority: GOOGLE_API_KEY -> OPENAI_API_KEY
        if os.environ.get("GOOGLE_API_KEY"):
            simulator_model = "gemini/gemini-1.5-flash"
            evaluation_model = "gemini/gemini-1.5-flash"
            print("[INFO] Using Gemini for attack simulation and evaluation")
        elif os.environ.get("OPENAI_API_KEY"):
            # Use gpt-4 which is available in most accounts
            simulator_model = "gpt-4"
            evaluation_model = "gpt-4"
            print("[INFO] Using OpenAI gpt-4 for attack simulation and evaluation")
        else:
            raise RuntimeError(
                "No API key found for attack simulation. "
                "Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        risk_assessment = await asyncio.to_thread(
            red_team,
            model_callback=callback,
            vulnerabilities=vulnerabilities,
            attacks=attacks,
            attacks_per_vulnerability_type=self.config.attacks_per_vulnerability,
            max_concurrent=self.config.max_concurrent,
            target_purpose=self.config.purpose,
            simulator_model=simulator_model,
            evaluation_model=evaluation_model,
            async_mode=False,  # Use sync mode with sync callback
        )

        # Parse results
        result = EvaluationResult(
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
            config=self.config,
        )

        # Access results via risk_assessment.overview.vulnerability_type_results
        if hasattr(risk_assessment, "overview") and risk_assessment.overview:
            for vuln_result in risk_assessment.overview.vulnerability_type_results:
                vuln_name = vuln_result.vulnerability
                passing = vuln_result.passing
                failing = vuln_result.failing
                total = passing + failing

                vr = VulnerabilityResult(
                    name=vuln_name,
                    attacks=total,
                    passed=passing,
                    failed=failing,
                )
                result.vulnerability_results.append(vr)
                result.total_attacks += total
                result.total_passed += passing
                result.total_failed += failing

        # Collect failed examples from test_cases
        if hasattr(risk_assessment, "test_cases") and risk_assessment.test_cases:
            for tc in risk_assessment.test_cases:
                if tc.score == 0 and len(result.failed_examples) < 10:
                    # Convert enum to string for JSON serialization
                    attack_type = "Unknown"
                    if hasattr(tc, "vulnerability_type") and tc.vulnerability_type:
                        vt = tc.vulnerability_type
                        attack_type = vt.value if hasattr(vt, "value") else str(vt)

                    result.failed_examples.append({
                        "vulnerability": tc.vulnerability if hasattr(tc, "vulnerability") else "Unknown",
                        "attack_type": attack_type,
                        "input": (tc.input[:500] if hasattr(tc, "input") and tc.input else ""),
                        "output": (tc.actual_output[:500] if hasattr(tc, "actual_output") and tc.actual_output else ""),
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
