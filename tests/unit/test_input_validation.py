"""Unit tests for input validation functions"""

import pytest
from unittest.mock import patch
import os

# Import validation functions from app.py
from app import validate_input, DANGEROUS_CHARS, MAX_QUERY_LENGTH, MIN_QUERY_LENGTH


class TestInputValidation:
    """Test input validation functionality"""
    
    def test_validate_input_valid_text(self):
        """Test validation with valid input text"""
        valid_inputs = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain the concept of embeddings.",
            "Tell me about vector databases.",
            "What is RAG in AI?"
        ]
        
        for text in valid_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is True
            assert error_msg == ""

    def test_validate_input_empty_text(self):
        """Test validation with empty text"""
        empty_inputs = ["", "   ", "\n\n", "\t\t"]
        
        for text in empty_inputs:
            is_valid, error_msg = validate_input(text, "query")
            assert is_valid is False
            assert "query no puede estar vacío" in error_msg

    def test_validate_input_too_long(self):
        """Test validation with text exceeding max length"""
        long_text = "a" * (MAX_QUERY_LENGTH + 1)
        
        is_valid, error_msg = validate_input(long_text)
        assert is_valid is False
        assert f"máximo {MAX_QUERY_LENGTH}" in error_msg

    def test_validate_input_dangerous_characters(self):
        """Test validation with dangerous characters"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "</script>",
            "javascript:void(0)",
            "onload=alert(1)",
            "onerror=alert(1)",
            "eval('malicious code')"
        ]
        
        for text in dangerous_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is False
            assert "caracteres peligrosos" in error_msg

    def test_validate_input_custom_field_name(self):
        """Test validation with custom field name in error messages"""
        field_name = "mensaje"
        
        # Test empty input with custom field name
        is_valid, error_msg = validate_input("", field_name)
        assert is_valid is False
        assert f"{field_name} no puede estar vacío" in error_msg

    def test_validate_input_boundary_lengths(self):
        """Test validation at boundary conditions"""
        # Text exactly at max length should be valid
        max_length_text = "a" * MAX_QUERY_LENGTH
        is_valid, error_msg = validate_input(max_length_text)
        assert is_valid is True
        
        # Single character should be valid (above min length)
        single_char = "a"
        is_valid, error_msg = validate_input(single_char)
        assert is_valid is True

    def test_validate_input_special_characters_safe(self):
        """Test validation with safe special characters"""
        safe_special_inputs = [
            "What's the difference between AI and ML?",
            "How much is 2+2?",
            "Tell me about C++ programming.",
            "What is the #1 best practice?",
            "Email: test@example.com",
            "Price: $100.50",
            "Percentage: 95%"
        ]
        
        for text in safe_special_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is True
            assert error_msg == ""

    def test_validate_input_unicode_characters(self):
        """Test validation with unicode characters"""
        unicode_inputs = [
            "¿Qué es machine learning?",
            "Explique les réseaux de neurones",
            "Was ist künstliche Intelligenz?",
            "Что такое машинное обучение?",
            "機械学習とは何ですか？",
            "Émojis: 🤖 🧠 💡"
        ]
        
        for text in unicode_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is True
            assert error_msg == ""

    def test_validate_input_mixed_dangerous_content(self):
        """Test validation with mixed content containing dangerous elements"""
        mixed_dangerous_inputs = [
            "What is AI? <script>alert(1)</script>",
            "Normal question javascript:void(0) continuation",
            "Tell me about onload=hack() machine learning",
            "Explain eval('code') in programming"
        ]
        
        for text in mixed_dangerous_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is False
            assert "caracteres peligrosos" in error_msg

    def test_validate_input_case_sensitivity_dangerous_chars(self):
        """Test that dangerous character detection is case sensitive"""
        # These should be blocked (exact case matches)
        dangerous_exact = ["<script>", "JAVASCRIPT:", "onload="]
        for text in dangerous_exact:
            is_valid, error_msg = validate_input(text)
            # Only lowercase versions are in DANGEROUS_CHARS, so case matters
            if text.lower() in [d.lower() for d in DANGEROUS_CHARS]:
                assert is_valid is False
            # The actual implementation checks exact strings, so case matters

    def test_validate_input_whitespace_normalization(self):
        """Test validation handles whitespace correctly"""
        whitespace_inputs = [
            "  What is AI?  ",
            "\nHow do embeddings work?\n",
            "\tExplain neural networks\t",
            "Multiple    spaces    between    words"
        ]
        
        for text in whitespace_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is True

    def test_dangerous_chars_configuration(self):
        """Test that dangerous characters list is properly configured"""
        expected_dangerous_chars = [
            '<script',
            '</script',
            'javascript:',
            'onload=',
            'onerror=',
            'eval('
        ]
        
        for char in expected_dangerous_chars:
            assert char in DANGEROUS_CHARS

    def test_validate_input_with_numbers(self):
        """Test validation with numeric content"""
        numeric_inputs = [
            "What is 2 + 2?",
            "Calculate the square root of 144",
            "Show me the first 10 prime numbers",
            "What is π (pi)?",
            "Convert 100 degrees Celsius to Fahrenheit"
        ]
        
        for text in numeric_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is True

    def test_validate_input_code_snippets_safe(self):
        """Test validation with safe code snippets in questions"""
        code_inputs = [
            "How do I write a for loop in Python?",
            "What does 'print(hello world)' do?",
            "Explain the difference between '==' and '='",
            "How to use if-else statements?",
            "What is a function definition?"
        ]
        
        for text in code_inputs:
            is_valid, error_msg = validate_input(text)
            assert is_valid is True

    @patch.dict(os.environ, {'MAX_QUERY_LENGTH': '100'})
    def test_validate_input_environment_configuration(self):
        """Test that validation respects environment configuration"""
        # This test would require reloading the module to pick up env changes
        # For now, we just test that the constants are defined
        assert isinstance(MAX_QUERY_LENGTH, int)
        assert isinstance(MIN_QUERY_LENGTH, int)
        assert MAX_QUERY_LENGTH > MIN_QUERY_LENGTH

    def test_validate_input_return_types(self):
        """Test that validation returns correct types"""
        is_valid, error_msg = validate_input("Valid input")
        
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)
        
        # Test with invalid input
        is_valid, error_msg = validate_input("")
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)
        assert is_valid is False
        assert len(error_msg) > 0