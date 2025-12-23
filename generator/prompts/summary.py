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
Answer using ONLY the regulatory excerpts below.
Do not add preambles. Do not mention the excerpts/context."""
    
    user = f"""Question category: {meta.get('category', 'unknown')}
Classification confidence: {meta.get('confidence', 0.0):.0%}

Regulatory context:
{context_text}

User question: {question}

HOUSE STYLE - CRITICAL:
- Do NOT say phrases like: "Based on the provided context", "According to the provided context", "From the context", "The context states", "The provided regulatory context".
- Do NOT mention that you were given context or that you are an AI.
- Start directly with the answer. Use plain declarative sentences.
- Use anchors/quotes as evidence, not as meta commentary.

Instructions:
1. Start with 1 sentence that answers the question directly.
2. Then add 1-3 sentences of clarification.
3. End with 1 anchor reference (e.g., §160.103).
4. Be specific and accurate.
5. If information is not found in the context, state "Insufficient context."

Answer:"""
    
    return system, user, 0.3, 300
