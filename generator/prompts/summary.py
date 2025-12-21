"""Prompt template for SUMMARY policy."""

from typing import List, Dict, Any, Tuple
from ..base import ContextItem


def build_prompt(
    question: str,
    context: List[ContextItem],
    meta: Dict[str, Any]
) -> Tuple[str, str, float, int]:
    """
    Builds prompt for summary/overview questions.
    
    Args:
        question: User question
        context: Context items from retriever
        meta: Metadata (category, confidence, etc.)
    
    Returns:
        Tuple of (system, user, temperature, max_tokens)
    """
    # Build context with full text
    context_parts = []
    for i, item in enumerate(context, 1):
        section_info = f"Section {item.section_number}: {item.section_title}"
        if item.anchor:
            section_info += f" ({item.anchor})"
        
        context_parts.append(
            f"[{i}] {section_info}\n{item.text_raw}"
        )
    
    context_text = "\n\n".join(context_parts)
    
    system = """You are an expert on HIPAA regulations.
Provide concise overviews based on the provided regulatory context.
Use only provided context. If insufficient, say insufficient context."""
    
    user = f"""Question category: {meta.get('category', 'unknown')}
Classification confidence: {meta.get('confidence', 0.0):.0%}

Regulatory context:
{context_text}

User question: {question}

Instructions:
1. Provide a concise overview (2-4 sentences) based on the provided context
2. Reference at least one anchor (e.g., §160.103) as a source
3. Be specific and accurate
4. Use only information from the provided context
5. If information is not found in the context, state "insufficient context"

Answer:"""
    
    return system, user, 0.3, 300
