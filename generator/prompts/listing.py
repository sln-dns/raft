"""Prompt template for LISTING policy."""

from typing import List, Dict, Any, Tuple
from ..base import ContextItem


def build_prompt(
    question: str,
    context: List[ContextItem],
    meta: Dict[str, Any]
) -> Tuple[str, str, float, int]:
    """
    Builds prompt for listing questions (scope, penalties, permissions).
    
    Args:
        question: User question
        context: Context items from retriever
        meta: Metadata (category, confidence, policy signals, etc.)
    
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
    
    # Check for permission policy signal (from retriever or determined)
    permission_policy = meta.get('permission_policy', '')
    retriever_policy_signal = meta.get('policy_signal', '')
    
    permission_instruction = ""
    if retriever_policy_signal:
        # Сигнал от ретривера - это подсказка, но LLM должен опираться на контекст
        permission_instruction = (
            f"\n6. Retriever policy signal: {retriever_policy_signal.upper()} "
            f"(permitted/conditional/prohibited/unclear). "
            f"This is a hint - verify with the provided context and reflect conditions/limitations in your answer."
        )
    elif permission_policy:
        # Fallback на определенный permission_policy
        permission_instruction = f"\n6. Policy signal: {permission_policy.upper()} - reflect this in your answer"
    
    system = """You are an expert on HIPAA regulations.
Answer questions by providing structured lists based on the provided regulatory context.
Use only provided context. If insufficient, say insufficient context.
You MUST return your answer in JSON format."""
    
    # Build list of available anchors for validation
    available_anchors = [item.anchor for item in context if item.anchor]
    anchors_list = ", ".join(available_anchors) if available_anchors else "none"
    
    user = f"""Question category: {meta.get('category', 'unknown')}
Classification confidence: {meta.get('confidence', 0.0):.0%}

Regulatory context:
{context_text}

User question: {question}

Available anchors (you can ONLY use these): {anchors_list}

Instructions:
1. Provide a structured list of items/conditions/requirements
2. Each item must include its anchor
3. You MUST return your response as JSON with this structure:
{{
  "answer": "structured list as text (each item with [anchor])",
  "citations": [
    {{"anchor": "§160.103", "quote": "exact quote from context"}},
    {{"anchor": "§164.502", "quote": "another exact quote"}}
  ]
}}
4. Use ONLY anchors from the available anchors list above
5. Quotes must be exact substrings from the corresponding context items
6. If information is not found in the context, set answer to "insufficient context" and citations to []
{permission_instruction}

Return ONLY valid JSON, no additional text:"""
    
    return system, user, 0.1, 1000
