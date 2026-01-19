# RedTeam

Red team evaluation for Hedgehog-trained models using [DeepTeam](https://github.com/confident-ai/deepteam).

## Installation

```bash
cd redteam
uv sync
```

You'll also need vLLM for model serving:

```bash
uv add vllm
```

## Usage

### Compare Base vs Trained Model

The primary use case - compare your Hedgehog-trained model against its base model:

```bash
# With automatic vLLM server management
redteam compare Qwen/Qwen2.5-3B-Instruct outputs/dai-model-merged \
    --output reports/comparison.json

# Quick test (fewer attacks, no multi-turn)
redteam compare Qwen/Qwen2.5-3B-Instruct outputs/dai-model-merged \
    --preset quick \
    --no-multi-turn

# Full evaluation (all vulnerability types)
redteam compare Qwen/Qwen2.5-3B-Instruct outputs/dai-model-merged \
    --preset full \
    --attacks 10

# Specify the simulator model explicitly
redteam compare Qwen/Qwen2.5-3B-Instruct outputs/dai-model-merged \
    --simulator-model gpt-4o
```

### With Pre-Started vLLM Servers

For faster iteration, start servers separately:

```bash
# Terminal 1: Base model
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000

# Terminal 2: Trained model
vllm serve outputs/dai-model-merged --port 8001

# Terminal 3: Run comparison
redteam compare Qwen/Qwen2.5-3B-Instruct outputs/dai-model-merged \
    --base-url http://localhost:8000 \
    --trained-url http://localhost:8001 \
    --output reports/comparison.json
```

### Evaluate Single Model

```bash
redteam evaluate outputs/dai-model-merged --output reports/trained.json
redteam evaluate Qwen/Qwen2.5-3B-Instruct --output reports/base.json
```

### Serve Model for Manual Testing

```bash
redteam serve outputs/dai-model-merged --port 8000
```

## Vulnerability Presets

| Preset | Vulnerabilities | Use Case |
|--------|-----------------|----------|
| `quick` | Shell Injection, Prompt Injection | Fast smoke test |
| `security` | Shell/SQL Injection, PII/Prompt Leakage, SSRF, Excessive Agency | Security-focused (default) |
| `full` | All above + Bias, Misinformation, Illegal Activity, etc. | Comprehensive red team |

## Attack Types

**Single-turn attacks:**
- Prompt Injection
- ROT13 encoding
- Leetspeak
- Math Problem

**Multi-turn attacks** (disable with `--no-multi-turn`):
- Linear Jailbreaking
- Tree Jailbreaking
- Crescendo

## Example Output

```
================================================================================
                    DEEPTEAM RED TEAM COMPARISON REPORT
================================================================================

Base Model: Qwen/Qwen2.5-3B-Instruct
Trained Model: outputs/dai-model-merged
Date: 2026-01-19T14:32:00Z

+---------------------------+
|    Overall Comparison     |
+---------------------------+
| Base Risk: 0.32  ->  Trained Risk: 0.08 |
| Improvement: +24%                        |
+---------------------------+

COMPARISON BY VULNERABILITY
--------------------------------------------------------------------------------
| Vulnerability        | Base Pass Rate | Trained Pass Rate | Improvement |
|---------------------|----------------|-------------------|-------------|
| ShellInjection      | 65%            | 92%               | +27%        |
| PIILeakage          | 70%            | 95%               | +25%        |
| PromptInjection     | 60%            | 88%               | +28%        |
| SSRF                | 75%            | 100%              | +25%        |
| ExcessiveAgency     | 55%            | 80%               | +25%        |
|---------------------|----------------|-------------------|-------------|
| Overall             | 68%            | 92%               | +24%        |
```

## JSON Report Structure

```json
{
  "generated_at": "2026-01-19T14:32:00Z",
  "base_model": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "risk_score": 0.32,
    "summary": {
      "total_attacks": 100,
      "total_passed": 68,
      "total_failed": 32,
      "pass_rate": 0.68
    },
    "vulnerability_results": [...]
  },
  "trained_model": {
    "model_name": "outputs/dai-model-merged",
    "risk_score": 0.08,
    "summary": {
      "total_attacks": 100,
      "total_passed": 92,
      "total_failed": 8,
      "pass_rate": 0.92
    },
    "vulnerability_results": [...]
  },
  "comparison": {
    "overall_improvement": 0.24,
    "improvements_by_vulnerability": {
      "ShellInjection": 0.27,
      "PIILeakage": 0.25
    }
  }
}
```

## Environment Variables

DeepTeam requires an LLM for attack simulation and evaluation. Set one of:

```bash
export OPENAI_API_KEY=sk-...          # OpenAI
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
export GOOGLE_API_KEY=AIza...         # Google
```

Or configure via CLI:

```bash
deepteam set-api-key sk-proj-abc123...
```

## Simulator Model

By default, the tool auto-detects which API key is set and uses an appropriate model:
- `GOOGLE_API_KEY` -> `gemini/gemini-1.5-flash`
- `OPENAI_API_KEY` -> `gpt-4o`

You can override this with `--simulator-model` (`-m`):

```bash
# Use a specific OpenAI model
redteam evaluate outputs/model --simulator-model gpt-4o

# Use Gemini
redteam evaluate outputs/model --simulator-model gemini/gemini-1.5-flash

# Use Claude (requires ANTHROPIC_API_KEY)
redteam evaluate outputs/model --simulator-model claude-3-5-sonnet-20241022
```
