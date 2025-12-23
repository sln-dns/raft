"""Smoke test для CitationRetriever scope parsing (Шаг 2.0).

Фиксирует проблему "было/стало":
- БЫЛО: CitationRetriever возвращал вперемешку §164.512(a)(b)(d)(e)(f)(j)...
- СТАЛО: CitationRetriever возвращает только §164.512(f) и его подпункты для law enforcement вопросов
"""

from retrievers.citation import CitationRetriever


def test_infer_anchor_prefix():
    """Тест функции infer_anchor_prefix для различных вопросов."""
    retriever = CitationRetriever()
    
    test_cases = [
        # Law enforcement - должен вернуть §164.512(f)
        (
            "Cite the HIPAA rule that addresses disclosures for law enforcement purposes.",
            "§164.512(f)"
        ),
        (
            "Cite where HIPAA allows disclosures to law enforcement and on what basis.",
            "§164.512(f)"
        ),
        # Suspect/fugitive - должен вернуть §164.512(f)(2)
        (
            "What are the provisions for disclosing PHI to law enforcement for identifying suspects?",
            "§164.512(f)(2)"
        ),
        (
            "Can HIPAA allow disclosure to locate a missing person?",
            "§164.512(f)(2)"
        ),
        # Victim + crime - должен вернуть §164.512(f)(3)
        (
            "Can HIPAA allow disclosure of PHI about crime victims?",
            "§164.512(f)(3)"
        ),
        # Явное указание anchor - должно уважаться
        (
            "What does §164.512(f)(2) say about suspects?",
            "§164.512(f)(2)"
        ),
    ]
    
    print("=== Тест infer_anchor_prefix ===")
    all_passed = True
    
    for question, expected_prefix in test_cases:
        actual_prefix = retriever.infer_anchor_prefix(question)
        passed = actual_prefix == expected_prefix
        status = "✅" if passed else "❌"
        
        print(f"{status} Q: {question[:60]}...")
        print(f"   Expected: {expected_prefix}")
        print(f"   Actual:   {actual_prefix}")
        
        if not passed:
            all_passed = False
        print()
    
    if all_passed:
        print("✅ Все тесты прошли")
    else:
        print("❌ Некоторые тесты не прошли")
    
    return all_passed


if __name__ == "__main__":
    test_infer_anchor_prefix()
