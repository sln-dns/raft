"""Тесты для классификации "colon type"."""

import unittest
from retrievers.citation import ends_with_colon, is_list_introducer, is_standard_header


class TestColonType(unittest.TestCase):
    """Тесты для функций классификации colon type."""
    
    def test_ends_with_colon(self):
        """Тест: ends_with_colon определяет двоеточие в конце"""
        self.assertTrue(ends_with_colon("(f) Standard:"))
        self.assertTrue(ends_with_colon("Permitted disclosures:"))
        self.assertTrue(ends_with_colon("Text with colon:  "))
        self.assertFalse(ends_with_colon("Text without colon"))
        self.assertFalse(ends_with_colon("Text with colon: in the middle."))
        print("✅ Test passed: ends_with_colon")
    
    def test_is_list_introducer(self):
        """Тест: is_list_introducer определяет вводы к списку"""
        self.assertTrue(is_list_introducer("(1) Permitted disclosures: ... may disclose protected health information:"))
        self.assertTrue(is_list_introducer("The following:"))
        self.assertTrue(is_list_introducer("As follows:"))
        self.assertTrue(is_list_introducer("Includes:"))
        self.assertTrue(is_list_introducer("Pursuant to the following:"))
        self.assertFalse(is_list_introducer("(f) Standard:"))
        self.assertFalse(is_list_introducer("Regular text without list introducer"))
        print("✅ Test passed: is_list_introducer")
    
    def test_is_standard_header(self):
        """Тест: is_standard_header определяет заголовочный Standard:"""
        self.assertTrue(is_standard_header("(f) Standard:"))
        self.assertTrue(is_standard_header("(a) Standard: Covered entities must comply."))
        self.assertFalse(is_standard_header("(1) Permitted disclosures: ... may disclose protected health information:"))
        self.assertFalse(is_standard_header("Regular text without Standard:"))
        self.assertFalse(is_standard_header("(f) Standard: ... may disclose protected health information:"))
        print("✅ Test passed: is_standard_header")
    
    def test_combined_cases(self):
        """Тест: комбинированные случаи"""
        # (f) Standard: ... -> standard_header=True
        self.assertTrue(is_standard_header("(f) Standard:"))
        self.assertFalse(is_list_introducer("(f) Standard:"))
        
        # (1) Permitted disclosures: ... may disclose protected health information: -> list_introducer=True
        text1 = "(1) Permitted disclosures: ... may disclose protected health information:"
        self.assertTrue(is_list_introducer(text1))
        self.assertFalse(is_standard_header(text1))
        
        # Обычная строка без : -> оба False
        text2 = "Regular text without colon"
        self.assertFalse(ends_with_colon(text2))
        self.assertFalse(is_list_introducer(text2))
        self.assertFalse(is_standard_header(text2))
        
        print("✅ Test passed: combined cases")


if __name__ == "__main__":
    unittest.main(verbosity=2)
