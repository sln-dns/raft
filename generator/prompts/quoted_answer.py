"""Prompt template for QUOTED_ANSWER policy."""

from typing import List, Dict, Any, Tuple
from ..base import ContextItem


def build_prompt(
    question: str,
    context: List[ContextItem],
    meta: Dict[str, Any]
) -> Tuple[str, str, float, int]:
    """
    Builds prompt for questions requiring quoted answers (definitions, procedural).
    
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
    
    # Check if this is a yes/no question (procedural)
    is_yesno = meta.get('category') == 'procedural / best practices'
    is_regulatory_principle = meta.get('category') == 'regulatory_principle'
    
    yesno_instruction = ""
    regulatory_principle_instruction = ""
    
    if is_yesno:
        # Используем сигнал от ретривера, если есть
        yesno_signal = meta.get('yesno_signal', '')
        yesno_rationale = meta.get('yesno_rationale', '')
        
        if yesno_signal:
            # Сигнал от ретривера - это подсказка, но LLM должен опираться на контекст
            rationale_text = f" (Retriever signal: {yesno_rationale})" if yesno_rationale else ""
            yesno_instruction = (
                f"\n6. Retriever signal: {yesno_signal.upper()}{rationale_text}. "
                f"Start your answer with: {yesno_signal.upper()} / NO / UNCLEAR "
                f"based on the evidence in the provided context (the signal is a hint, verify with context)."
            )
        else:
            yesno_instruction = "\n6. Start your answer with: YES / NO / UNCLEAR (based on the evidence in context)"
    
    # Special handling for regulatory_principle questions
    if is_regulatory_principle:
        # Extract the term/concept from the question if available
        question_lower = question.lower()
        term_match = None
        import re
        # Try to extract term from common patterns
        for pattern in [
            r'what does\s+["\']([^"\']+)["\']\s+mean',
            r'what does\s+([A-Za-z\s]+?)\s+mean',
            r'define\s+([A-Za-z\s]+?)(?:\s|$)',
            r'what is\s+([A-Za-z\s]+?)(?:\s|$)',
        ]:
            match = re.search(pattern, question_lower, re.IGNORECASE)
            if match:
                term_match = match.group(1).strip()
                break
        
        term = term_match if term_match else "the concept"
        regulatory_principle_instruction = (
            f"\n6. SPECIAL INSTRUCTIONS FOR REGULATORY PRINCIPLE:\n"
            f"   - Start with: \"HIPAA does not provide a standalone definition of '{term}' in the Definitions section.\"\n"
            f"   - Explain the regulatory meaning as a requirement/principle, not as a dictionary definition.\n"
            f"   - DO NOT invent a definition like \"{term} means...\" if the context does not contain a formal definition.\n"
            f"   - Instead, describe what HIPAA requires or how the principle applies based on the provided regulatory text.\n"
            f"   - You MUST include 1-3 exact quotes with anchors from the context to support your explanation."
        )
    
    system = """You are an expert on HIPAA regulations.
Answer questions using exact quotes from the provided regulatory context.
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
1. Provide a brief answer (1-2 sentences)
2. Include 1-3 exact quotes from the context with their anchors
3. You MUST return your response as JSON with this structure:
{{
  "answer": "your brief answer text",
  "citations": [
    {{"anchor": "§160.103", "quote": "exact quote text from context"}},
    {{"anchor": "§164.502", "quote": "another exact quote"}}
  ]
}}
4. Use ONLY anchors from the available anchors list above
5. Quotes must be exact substrings from the corresponding context items
6. If information is not found in the context, set answer to "insufficient context" and citations to []
{yesno_instruction}
{regulatory_principle_instruction}

Return ONLY valid JSON, no additional text:"""
    
    return system, user, 0.1, 600
