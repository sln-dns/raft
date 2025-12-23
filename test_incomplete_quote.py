"""Тесты для детектора неполных цитат."""

import unittest
from generator.citation_validator import is_incomplete_quote


class TestIncompleteQuote(unittest.TestCase):
    """Тесты для функции is_incomplete_quote()."""
    
    def test_ends_with_colon(self):
        """Тест: текст, заканчивающийся на ':' должен быть incomplete"""
        self.assertTrue(is_incomplete_quote("Permitted uses and disclosures:"))
        self.assertTrue(is_incomplete_quote("A covered entity may disclose protected health information:"))
        self.assertTrue(is_incomplete_quote("Text with colon at end:  "))
        print("✅ Test passed: ends with colon -> incomplete")
    
    def test_does_not_end_with_colon(self):
        """Тест: текст, НЕ заканчивающийся на ':' НЕ должен быть incomplete"""
        self.assertFalse(is_incomplete_quote("A covered entity may disclose protected health information."))
        self.assertFalse(is_incomplete_quote("This is a complete sentence."))
        self.assertFalse(is_incomplete_quote("Text without colon"))
        print("✅ Test passed: does not end with colon -> not incomplete")
    
    def test_empty_string(self):
        """Тест: пустая строка НЕ должна быть incomplete"""
        self.assertFalse(is_incomplete_quote(""))
        self.assertFalse(is_incomplete_quote("   "))
        print("✅ Test passed: empty string -> not incomplete")
    
    def test_incomplete_patterns(self):
        """Тест: опциональные паттерны incomplete"""
        self.assertTrue(is_incomplete_quote("may disclose protected health information:"))
        self.assertTrue(is_incomplete_quote("may use or disclose protected health information:"))
        self.assertTrue(is_incomplete_quote("permitted uses and disclosures:"))
        print("✅ Test passed: incomplete patterns detected")
    
    def test_complete_sentences_with_colon_in_middle(self):
        """Тест: полные предложения с двоеточием в середине НЕ должны быть incomplete"""
        self.assertFalse(is_incomplete_quote("The regulation states: covered entities must comply."))
        self.assertFalse(is_incomplete_quote("HIPAA requires: minimum necessary standard."))
        print("✅ Test passed: colon in middle -> not incomplete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
