"""Token cost estimation for LLM API calls.

Hardcoded rates for MVP. These will be configurable in a
future version. Rates are per 1,000 tokens.
"""

from __future__ import annotations

from sylo.models import TokenUsage

# Cost per 1,000 tokens (input and output rates)
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.00025, "output": 0.00125},
}

# Fallback rate when model is unknown
DEFAULT_COST_PER_1K: dict[str, float] = {"input": 0.002, "output": 0.008}


def estimate_cost(usage: TokenUsage) -> float:
    """Estimate the USD cost for a given token usage.

    Uses model-specific rates when available, otherwise falls back
    to a reasonable default rate.

    Args:
        usage: Token usage record with prompt/completion counts and model name.

    Returns:
        Estimated cost in USD.
    """
    rates = COST_PER_1K_TOKENS.get(usage.model or "", DEFAULT_COST_PER_1K)

    input_cost = (usage.prompt_tokens / 1000) * rates["input"]
    output_cost = (usage.completion_tokens / 1000) * rates["output"]

    return round(input_cost + output_cost, 6)


def extract_token_usage(result: dict) -> TokenUsage | None:
    """Extract token usage from a step's return value.

    Looks for a "usage" key following the standard OpenAI/Anthropic format:
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "model": "gpt-4o"
            }
        }

    Also checks for "token_usage" key as an alternative.

    Args:
        result: The step function's return value (must be a dict).

    Returns:
        TokenUsage if usage data was found, None otherwise.
    """
    if not isinstance(result, dict):
        return None

    usage_data = result.get("usage") or result.get("token_usage")
    if not isinstance(usage_data, dict):
        return None

    prompt_tokens = usage_data.get("prompt_tokens", 0)
    completion_tokens = usage_data.get("completion_tokens", 0)
    total_tokens = usage_data.get(
        "total_tokens", prompt_tokens + completion_tokens
    )
    model = usage_data.get("model")

    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=model,
    )

    # Calculate estimated cost
    usage.estimated_cost_usd = estimate_cost(usage)

    return usage
