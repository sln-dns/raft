"""Prompt template for NAVIGATION policy."""

from typing import List, Dict, Any, Tuple
from ..base import ContextItem


def build_prompt(
    question: str,
    context: List[ContextItem],
    meta: Dict[str, Any]
) -> Tuple[str, str, float, int]:
    """
    Builds prompt for navigation questions (document structure).
    
    Args:
        question: User question
        context: Context items from retriever
        meta: Metadata (category, confidence, etc.)
    
    Returns:
        Tuple of (system, user, temperature, max_tokens)
    """
    # Build context - only section structure, no text quotes
    context_parts = []
    for i, item in enumerate(context, 1):
        section_info = f"Section {item.section_number}: {item.section_title}"
        if item.anchor:
            section_info += f" ({item.anchor})"
        context_parts.append(f"[{i}] {section_info}")
    
    context_text = "\n".join(context_parts)
    
    system = """You are an expert on HIPAA regulations document structure.
Answer questions about which parts/sections cover specific topics.
Do NOT quote full text - only reference section numbers and anchors.
Use only provided context. If insufficient, say insufficient context."""
    
    user = f"""Question category: {meta.get('category', 'unknown')}
Classification confidence: {meta.get('confidence', 0.0):.0%}

Document structure:
{context_text}

User question: {question}

Instructions:
1. Identify which part/section covers the requested topic
2. Be specific - provide section numbers and their titles
3. Reference anchors (e.g., §160.103) if available
4. Do NOT quote full text from regulations
5. If information is not found in the context, state "insufficient context"

Answer:"""
    
    return system, user, 0.2, 500
