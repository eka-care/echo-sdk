"""
Claude model capability detection.

Anthropic's request surface changed across model generations, so the same
payload is not valid for every Claude model:

- Claude Sonnet 5 / Opus 5 / Fable 5 / Mythos 5 and Opus 4.7+ reject
  ``temperature`` / ``top_p`` / ``top_k`` and the legacy
  ``{"type": "enabled", "budget_tokens": N}`` thinking form with a 400.
- Those models take ``{"type": "adaptive"}`` plus ``output_config.effort``
  instead.
- The 5-series additionally runs adaptive thinking when ``thinking`` is
  omitted, where Sonnet 4.6 and older ran with thinking off.
- Fable 5 / Mythos 5 reject ``{"type": "disabled"}`` outright — thinking is
  always on there.

Providers ask this module what a model accepts rather than substring-matching
model IDs at the call site. Model IDs are recognised in both first-party form
(``claude-sonnet-5``) and Bedrock form (``us.anthropic.claude-sonnet-5-v1:0``).
Anything unrecognised — a non-Claude model, or a naming scheme newer than this
module — falls back to the pre-5 surface, which is the safe default: it is what
every model accepted before Opus 4.7.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

# `claude-sonnet-4-5-20250929`, `anthropic.claude-opus-4-8`, `claude-fable-5`
_MODERN_ID = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?")
# `claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022`
_LEGACY_ID = re.compile(r"claude-(\d+)(?:-(\d+))?-(opus|sonnet|haiku)")


@dataclass(frozen=True)
class ClaudeCapabilities:
    """What a given Claude model accepts on the Messages API."""

    #: ``temperature`` / ``top_p`` / ``top_k`` are accepted (400 otherwise).
    accepts_sampling_params: bool
    #: ``thinking={"type": "enabled", "budget_tokens": N}`` is accepted.
    accepts_budget_tokens: bool
    #: ``thinking={"type": "adaptive"}`` is accepted.
    supports_adaptive_thinking: bool
    #: Omitting ``thinking`` entirely turns thinking *on*.
    thinking_on_by_default: bool
    #: ``thinking={"type": "disabled"}`` is accepted.
    can_disable_thinking: bool
    #: ``output_config.effort`` is accepted.
    supports_effort: bool


# Everything Anthropic shipped before Opus 4.7, plus every non-Claude model.
_PRE_5_SURFACE = ClaudeCapabilities(
    accepts_sampling_params=True,
    accepts_budget_tokens=False,
    supports_adaptive_thinking=False,
    thinking_on_by_default=False,
    can_disable_thinking=True,
    supports_effort=False,
)


def _parse(model: str) -> Optional[Tuple[str, int, int]]:
    """Extract ``(family, major, minor)`` from a Claude model ID, if it is one."""
    match = _MODERN_ID.search(model)
    if match:
        family, major, minor = match.group(1), match.group(2), match.group(3)
    else:
        match = _LEGACY_ID.search(model)
        if not match:
            return None
        major, minor, family = match.group(1), match.group(2), match.group(3)
    return family, int(major), int(minor or 0)


@lru_cache(maxsize=128)
def claude_capabilities(model: str) -> ClaudeCapabilities:
    """
    Resolve the request surface for ``model``.

    Cached because providers call it per request and the answer is a pure
    function of the model ID.
    """
    parsed = _parse(model or "")
    if parsed is None:
        return _PRE_5_SURFACE
    family, major, minor = parsed

    # Opus 4.7 was the cut-over: sampling params and `budget_tokens` removed,
    # adaptive thinking became the only thinking form. The whole 5-series
    # inherits that surface.
    next_gen = major >= 5
    strict = next_gen or (family == "opus" and major == 4 and minor >= 7)

    # Adaptive thinking arrived on Opus 4.6 / Sonnet 4.6. Haiku 4.5 and the
    # 4.0/4.5 tier only ever had the budget form.
    adaptive = strict or (family in ("opus", "sonnet") and major == 4 and minor >= 6)
    # Extended thinking (budget form) is a Claude 4 feature; it still works on
    # 4.6, and is gone from 4.7 onwards.
    budget = not strict and major == 4 and family in ("opus", "sonnet", "haiku")

    # Effort went GA on Opus 4.5, then across the 4.6 tier and everything after.
    effort = strict or (major == 4 and (minor >= 6 or (family == "opus" and minor >= 5)))

    return ClaudeCapabilities(
        accepts_sampling_params=not strict,
        accepts_budget_tokens=budget,
        supports_adaptive_thinking=adaptive,
        # Opus 4.7/4.8 still default to no thinking; only the 5-series flipped.
        thinking_on_by_default=next_gen,
        # Fable 5 / Mythos 5 think unconditionally and 400 on an explicit
        # `disabled`; the param has to be omitted there.
        can_disable_thinking=family not in ("fable", "mythos"),
        supports_effort=effort,
    )
