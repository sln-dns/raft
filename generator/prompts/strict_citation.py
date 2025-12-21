"""Prompt template for STRICT_CITATION policy.

Note: STRICT_CITATION policy should not call LLM (see step 7).
This module is prepared for future use.
"""

from typing import List, Dict, Any, Tuple
from ..base import ContextItem


def build_prompt(
    question: str,
    context: List[ContextItem],
    meta: Dict[str, Any]
) -> Tuple[str, str, float, int]:
    """
    Builds prompt for strict citation policy.
    
    Note: This policy should return raw citations without LLM processing.
    This function is a placeholder for future implementation.
    
    Args:
        question: User question
        context: Context items from retriever
        meta: Metadata (category, confidence, etc.)
    
    Returns:
        Tuple of (system, user, temperature, max_tokens)
    """
    # For STRICT_CITATION, we should not use LLM
    # This will be handled in step 7
    system = "You are an expert on HIPAA regulations."
    user = f"Question: {question}\n\nProvide exact quotes from the context with anchors."
    return system, user, 0.1, 1000
