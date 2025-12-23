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
Answer using ONLY the document structure below.
Do not add preambles. Do not mention the structure/context.
Do NOT quote full text - only reference section numbers and anchors."""
    
    user = f"""Question category: {meta.get('category', 'unknown')}
Classification confidence: {meta.get('confidence', 0.0):.0%}

Document structure:
{context_text}

User question: {question}

HOUSE STYLE - CRITICAL:
- Do NOT say phrases like: "Based on the provided context", "According to the provided context", "From the context", "The context states", "The provided regulatory context".
- Do NOT mention that you were given context or that you are an AI.
- Start directly with the answer. Use plain declarative sentences.
- Use anchors/quotes as evidence, not as meta commentary.

Instructions:
1. Start directly with which part/section covers the requested topic.
2. Be specific - provide section numbers and their titles.
3. Reference anchors (e.g., §160.103) if available.
4. Do NOT quote full text from regulations.
5. If information is not found in the context, state "Insufficient context."

Answer:"""
    
    return system, user, 0.2, 500
