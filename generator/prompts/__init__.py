"""Prompt templates for different answer policies."""

from .strict_citation import build_prompt as build_strict_citation_prompt
from .navigation import build_prompt as build_navigation_prompt
from .summary import build_prompt as build_summary_prompt
from .quoted_answer import build_prompt as build_quoted_answer_prompt
from .listing import build_prompt as build_listing_prompt

__all__ = [
    "build_strict_citation_prompt",
    "build_navigation_prompt",
    "build_summary_prompt",
    "build_quoted_answer_prompt",
    "build_listing_prompt",
]
