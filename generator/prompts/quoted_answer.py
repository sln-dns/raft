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
                f"\n8. Retriever signal: {yesno_signal.upper()}{rationale_text}. "
                f"Start your answer with: {yesno_signal.upper()} / NO / UNCLEAR "
                f"(the signal is a hint, verify with the excerpts above)."
            )
        else:
            yesno_instruction = "\n8. Start your answer with: YES / NO / UNCLEAR (verify with the excerpts above)"
    
    # Special handling for regulatory_principle questions
    if is_regulatory_principle:
        # Use concept_term from retriever_signals if available (from classification override)
        term = meta.get('concept_term')
        if not term:
            # Fallback: Extract the term/concept from the question if available
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
        
        # CRITICAL: For regulatory_principle, "not formally defined" is a VALID answer, not "insufficient context"
        regulatory_principle_instruction = (
            f"\n6. CRITICAL INSTRUCTIONS FOR REGULATORY PRINCIPLE (THIS IS NOT A DEFINITION QUESTION):\n"
            f"   - You MUST start your answer with: \"HIPAA does not provide a standalone definition of '{term}' in the Definitions section.\"\n"
            f"   - This is CORRECT and EXPECTED - do NOT say \"insufficient context\" for regulatory principles.\n"
            f"   - After the opening statement, explain '{term}' as a regulatory requirement/principle, NOT as a dictionary definition.\n"
            f"   - DO NOT write \"{term} means...\" or \"{term} is defined as...\" - instead describe what HIPAA requires or how the principle applies.\n"
            f"   - You MUST include 1-3 exact quotes with anchors from the provided context to support your explanation.\n"
            f"   - The quotes should show how '{term}' is used as a requirement or standard in the regulatory text.\n"
            f"   - Example format: \"HIPAA does not provide a standalone definition of '{term}' in the Definitions section. "
            f"However, §164.502(b) establishes it as a standard requiring covered entities to... [quote from context].\""
        )
    
    system = """You are an expert on HIPAA regulations.
Answer using ONLY the regulatory excerpts below.
Do not add preambles. Do not mention the excerpts/context.
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

HOUSE STYLE - CRITICAL:
- Do NOT say phrases like: "Based on the provided context", "According to the provided context", "From the context", "The context states", "The provided regulatory context".
- Do NOT mention that you were given context or that you are an AI.
- Start directly with the answer. Use plain declarative sentences.
- Use anchors/quotes as evidence, not as meta commentary.

Instructions:
1. Answer in 1-2 sentences. The first sentence must be the conclusion.
2. Then provide 1-3 exact quotes with anchors (you may use quotes from different sections/anchors)
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
6. The "answer" field MUST NOT contain meta phrases (context/disclaimer). It must contain only the user-facing answer.
7. If insufficient, return "answer": "Insufficient context." (without explanations why).
{yesno_instruction}
{regulatory_principle_instruction}
9. IMPORTANT: Only use "insufficient context" if the provided context is completely irrelevant to the question. 
   For regulatory_principle questions, "not formally defined" is a VALID answer - do NOT use "insufficient context".

Return ONLY valid JSON, no additional text:"""
    
    return system, user, 0.1, 600
