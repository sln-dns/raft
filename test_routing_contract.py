"""Тесты для проверки route contract - правильности работы apply_classification_override() и роутинга."""

import unittest
from unittest.mock import Mock, patch
from classification_override import apply_classification_override
from classifier import QuestionClassification


class TestRoutingContract(unittest.TestCase):
    """Тесты для проверки route contract после apply_classification_override()."""
    
    def test_1_minimum_necessary_regulatory_principle(self):
        """Тест 1: minimum necessary -> category=regulatory_principle, citation_mode != strict"""
        question = "What does minimum necessary mean?"
        
        # Мокаем исходную классификацию (LLM вернул definition)
        initial_classification = QuestionClassification(
            category="definition",
            confidence=0.85,
            reasoning="Question asks for definition of a term"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.category, "regulatory_principle", 
                        "minimum necessary should be classified as regulatory_principle")
        self.assertNotEqual(result.citation_mode, "strict",
                           "citation_mode should NOT be strict for regulatory_principle without explicit cite")
        self.assertEqual(concept_term, "minimum necessary",
                        "concept_term should be extracted")
        print("✅ Test 1 passed: minimum necessary -> regulatory_principle")
    
    def test_2_explicit_anchor_citation_required(self):
        """Тест 2: 'Cite §164.512(f)(2) …' -> category=citation-required, citation_mode=strict, anchor_hint=§164.512(f)(2)"""
        question = "Cite §164.512(f)(2) regarding disclosures to identify suspects."
        
        initial_classification = QuestionClassification(
            category="other",
            confidence=0.8,
            reasoning="Question asks to cite a specific section"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.category, "citation-required",
                        "Explicit anchor should force citation-required")
        self.assertEqual(result.citation_mode, "strict",
                        "Explicit anchor should force strict citation_mode")
        self.assertEqual(result.anchor_hint, "§164.512(f)(2)",
                        "anchor_hint should be extracted and normalized")
        self.assertTrue(result.require_citations,
                       "require_citations should be True")
        print("✅ Test 2 passed: explicit anchor -> citation-required + strict")
    
    def test_3_suspect_cite_strict_f2(self):
        """Тест 3: '… suspect … Cite.' -> strict, anchor_hint f(2)"""
        question = "Show the exact text regarding disclosures to identify suspects. Cite."
        
        initial_classification = QuestionClassification(
            category="other",
            confidence=0.8,
            reasoning="Question about suspect identification"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.citation_mode, "strict",
                        "Cite + suspect should result in strict citation_mode")
        self.assertEqual(result.category, "citation-required",
                        "Cite + suspect should result in citation-required")
        self.assertEqual(result.anchor_hint, "§164.512(f)(2)",
                        "suspect should map to anchor_hint §164.512(f)(2)")
        self.assertEqual(result.scope_hint, "suspect/fugitive/witness/missing person",
                        "suspect should map to correct scope_hint")
        print("✅ Test 3 passed: suspect + cite -> strict + f(2)")
    
    def test_4_victim_crime_cite_strict_f3(self):
        """Тест 4: '… victim of a crime … Cite.' -> strict, anchor_hint f(3)"""
        question = "Cite the regulation about disclosure of PHI about crime victims."
        
        initial_classification = QuestionClassification(
            category="other",
            confidence=0.8,
            reasoning="Question about crime victims"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.citation_mode, "strict",
                        "Cite + victim + crime should result in strict citation_mode")
        self.assertEqual(result.category, "citation-required",
                        "Cite + victim + crime should result in citation-required")
        self.assertEqual(result.anchor_hint, "§164.512(f)(3)",
                        "victim + crime should map to anchor_hint §164.512(f)(3)")
        self.assertEqual(result.scope_hint, "crime victim",
                        "victim + crime should map to correct scope_hint")
        print("✅ Test 4 passed: victim + crime + cite -> strict + f(3)")
    
    def test_5_documentation_cite_quoted_not_required(self):
        """Тест 5: 'How long retain documentation? Cite.' -> NOT citation-required, but quoted"""
        question = "How long retain documentation? Cite."
        
        initial_classification = QuestionClassification(
            category="procedural / best practices",
            confidence=0.8,
            reasoning="Question about documentation retention"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertNotEqual(result.category, "citation-required",
                           "Cite without strict topic should NOT change category to citation-required")
        self.assertEqual(result.citation_mode, "quoted",
                        "Cite without strict topic should result in quoted citation_mode")
        self.assertTrue(result.require_citations,
                       "require_citations should be True")
        self.assertEqual(result.scope_hint, "cite_requested",
                        "scope_hint should be cite_requested")
        print("✅ Test 5 passed: documentation + cite -> quoted, NOT citation-required")
    
    def test_6_define_business_associate_cite_quoted(self):
        """Тест 6: 'Define business associate. Cite.' -> definition + quoted"""
        question = "Define business associate. Cite."
        
        initial_classification = QuestionClassification(
            category="definition",
            confidence=0.9,
            reasoning="Question asks for definition of a term"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.category, "definition",
                        "category should remain definition")
        self.assertEqual(result.citation_mode, "quoted",
                        "definition + cite should result in quoted citation_mode")
        self.assertTrue(result.require_citations,
                       "require_citations should be True")
        print("✅ Test 6 passed: define + cite -> definition + quoted")
    
    def test_7_overall_purpose_none(self):
        """Тест 7: 'overall purpose…' -> overview/purpose + none"""
        question = "What is the overall purpose of HIPAA regulations?"
        
        initial_classification = QuestionClassification(
            category="overview / purpose",
            confidence=0.9,
            reasoning="Question asks about overall purpose"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.category, "overview / purpose",
                        "category should remain overview / purpose")
        self.assertEqual(result.citation_mode, "none",
                        "overview/purpose without cite should have citation_mode=none")
        self.assertFalse(result.require_citations,
                        "require_citations should be False")
        print("✅ Test 7 passed: overall purpose -> overview/purpose + none")
    
    def test_8_ephi_transmission_cite_quoted_not_strict(self):
        """Тест 8: 'Does HIPAA mention protecting ePHI during transmission? Cite.' -> procedural/other + quoted (не strict)"""
        question = "Does HIPAA mention protecting ePHI during transmission? Cite."
        
        initial_classification = QuestionClassification(
            category="procedural / best practices",
            confidence=0.85,
            reasoning="Question about ePHI protection procedures"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertNotEqual(result.citation_mode, "strict",
                           "Cite without strict topic should NOT be strict")
        self.assertEqual(result.citation_mode, "quoted",
                        "Cite without strict topic should be quoted")
        self.assertNotEqual(result.category, "citation-required",
                           "category should NOT be citation-required")
        self.assertTrue(result.require_citations,
                       "require_citations should be True")
        print("✅ Test 8 passed: ePHI + cite -> quoted, NOT strict")
    
    def test_9_law_enforcement_cite_strict_f(self):
        """Тест 9: 'Cite the HIPAA rule for law enforcement purposes' -> citation-required + strict + f"""
        question = "Cite the HIPAA rule that addresses disclosures for law enforcement purposes."
        
        initial_classification = QuestionClassification(
            category="other",
            confidence=0.8,
            reasoning="Question about law enforcement"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.category, "citation-required",
                        "Cite + law enforcement should result in citation-required")
        self.assertEqual(result.citation_mode, "strict",
                        "Cite + law enforcement should result in strict citation_mode")
        self.assertEqual(result.anchor_hint, "§164.512(f)",
                        "law enforcement should map to anchor_hint §164.512(f)")
        self.assertEqual(result.scope_hint, "law enforcement",
                        "law enforcement should map to correct scope_hint")
        print("✅ Test 9 passed: law enforcement + cite -> strict + f")
    
    def test_10_explicit_anchor_with_spaces_normalized(self):
        """Тест 10: explicit anchor with spaces '§ 164.512(f)' -> normalized"""
        question = "What does § 164.512(f) say about law enforcement?"
        
        initial_classification = QuestionClassification(
            category="other",
            confidence=0.8,
            reasoning="Question with explicit anchor"
        )
        
        result, concept_term = apply_classification_override(initial_classification, question)
        
        # Проверки
        self.assertEqual(result.category, "citation-required",
                        "Explicit anchor should force citation-required")
        self.assertEqual(result.citation_mode, "strict",
                        "Explicit anchor should force strict citation_mode")
        # Проверяем нормализацию: пробелы должны быть убраны
        self.assertEqual(result.anchor_hint, "§164.512(f)",
                        "anchor_hint should be normalized (spaces removed after §)")
        self.assertNotIn(" ", result.anchor_hint,
                        "anchor_hint should not contain spaces")
        print("✅ Test 10 passed: explicit anchor with spaces -> normalized")


if __name__ == "__main__":
    # Настройка логирования для тестов
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Запуск тестов
    unittest.main(verbosity=2)
