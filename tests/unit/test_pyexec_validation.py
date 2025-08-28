"""Unit tests for pyexec AST validation"""

import pytest
import ast
from unittest.mock import patch, Mock

# Import the functions to test
# We'll need to extract these functions or import them properly
from pyexec_service import (
    validate_ast,
    validate_expression_length,
    validate_expression_complexity,
    ValidationError,
    ErrorCategory,
    MAX_EXPR_LENGTH,
    MAX_EXPR_COMPLEXITY
)


class TestPyexecValidation:
    """Test AST validation functionality in pyexec service"""
    
    def test_validate_expression_length_valid(self):
        """Test validation of valid expression length"""
        short_expr = "1 + 2"
        # Should not raise exception
        validate_expression_length(short_expr)

    def test_validate_expression_length_too_long(self):
        """Test validation of overly long expression"""
        long_expr = "1 + 2" * (MAX_EXPR_LENGTH // 3)  # Create long expression
        
        with pytest.raises(ValidationError) as exc_info:
            validate_expression_length(long_expr)
        
        assert exc_info.value.category == ErrorCategory.TOO_LONG
        assert "demasiado larga" in str(exc_info.value)

    def test_validate_expression_complexity_simple(self):
        """Test validation of simple expression complexity"""
        simple_expr = ast.parse("1 + 2", mode="eval")
        # Should not raise exception
        validate_expression_complexity(simple_expr)

    def test_validate_expression_complexity_too_complex(self):
        """Test validation of overly complex expression"""
        # Create a complex nested expression
        complex_expr_str = " + ".join(["1"] * (MAX_EXPR_COMPLEXITY + 10))
        complex_expr = ast.parse(complex_expr_str, mode="eval")
        
        with pytest.raises(ValidationError) as exc_info:
            validate_expression_complexity(complex_expr)
        
        assert exc_info.value.category == ErrorCategory.TOO_COMPLEX
        assert "demasiado compleja" in str(exc_info.value)

    def test_validate_ast_valid_expression(self):
        """Test AST validation with valid mathematical expression"""
        valid_expressions = [
            "1 + 2",
            "3 * 4 - 5",
            "sqrt(16)",
            "abs(-10)",
            "min(1, 2, 3)",
            "max(4, 5, 6)",
            "pow(2, 3)",
            "sum([1, 2, 3])"
        ]
        
        for expr in valid_expressions:
            # Should not raise exception
            validate_ast(expr)

    def test_validate_ast_syntax_error(self):
        """Test AST validation with syntax error"""
        invalid_expressions = [
            "1 +",
            "* 2",
            "1 + + 2",
            "((1 + 2)",
            "1 + 2)",
            ""
        ]
        
        for expr in invalid_expressions:
            with pytest.raises(ValidationError) as exc_info:
                validate_ast(expr)
            assert exc_info.value.category == ErrorCategory.SYNTAX_ERROR

    def test_validate_ast_forbidden_nodes(self):
        """Test AST validation with forbidden node types"""
        forbidden_expressions = [
            "import os",  # Import statement
            "exec('print(1)')",  # Exec call
            "eval('1+1')",  # Eval call
            "lambda x: x + 1",  # Lambda
            "x = 1",  # Assignment
            "[x for x in range(10)]",  # List comprehension
        ]
        
        for expr in forbidden_expressions:
            with pytest.raises(ValidationError) as exc_info:
                validate_ast(expr)
            assert exc_info.value.category in [ErrorCategory.FORBIDDEN_NODE, ErrorCategory.SYNTAX_ERROR]

    def test_validate_ast_forbidden_functions(self):
        """Test AST validation with forbidden function calls"""
        forbidden_functions = [
            "print('hello')",
            "open('file.txt')",
            "input('Enter:')",
            "__import__('os')",
            "globals()",
            "locals()",
            "dir()",
            "vars()",
        ]
        
        for expr in forbidden_functions:
            with pytest.raises(ValidationError) as exc_info:
                validate_ast(expr)
            assert exc_info.value.category == ErrorCategory.FORBIDDEN_FUNCTION

    def test_validate_ast_allowed_math_functions(self):
        """Test AST validation with allowed mathematical functions"""
        allowed_math_expressions = [
            "sin(3.14159)",
            "cos(0)",
            "tan(1)",
            "log(10)",
            "sqrt(25)",
            "ceil(3.14)",
            "floor(3.14)",
            "factorial(5)",
            "gcd(12, 8)",
            "mean([1, 2, 3, 4, 5])",
            "median([1, 2, 3, 4, 5])",
            "stdev([1, 2, 3, 4, 5])"
        ]
        
        for expr in allowed_math_expressions:
            # Should not raise exception
            validate_ast(expr)

    def test_validate_ast_allowed_built_in_functions(self):
        """Test AST validation with allowed built-in functions"""
        allowed_builtin_expressions = [
            "abs(-5)",
            "min(1, 2, 3)",
            "max(4, 5, 6)",
            "sum([1, 2, 3])",
            "len([1, 2, 3, 4])",
            "round(3.14159, 2)",
            "pow(2, 8)",
            "divmod(10, 3)",
            "int(3.14)",
            "float(5)",
            "bool(1)"
        ]
        
        for expr in allowed_builtin_expressions:
            # Should not raise exception
            validate_ast(expr)

    def test_validate_ast_complex_valid_expression(self):
        """Test AST validation with complex but valid expression"""
        complex_valid = "sqrt(pow(3, 2) + pow(4, 2)) + abs(min(-1, -2, -3))"
        # Should not raise exception
        validate_ast(complex_valid)

    def test_validate_ast_nested_operations(self):
        """Test AST validation with nested mathematical operations"""
        nested_expressions = [
            "((1 + 2) * 3) / 4",
            "sqrt(abs(-16))",
            "pow(sin(0), 2) + pow(cos(0), 2)",
            "min(max(1, 2), max(3, 4))",
            "round(sum([1.1, 2.2, 3.3]), 2)"
        ]
        
        for expr in nested_expressions:
            # Should not raise exception
            validate_ast(expr)

    def test_validate_ast_edge_cases(self):
        """Test AST validation with edge cases"""
        edge_cases = [
            "0",
            "3.14159",
            "-42",
            "1e10",
            "1.5e-3",
            "True",
            "False"
        ]
        
        for expr in edge_cases:
            # Should not raise exception
            validate_ast(expr)

    def test_validate_ast_whitespace_handling(self):
        """Test AST validation handles whitespace correctly"""
        expressions_with_whitespace = [
            "  1 + 2  ",
            "\n1 * 2\n",
            "\t3 / 4\t",
            "   sqrt(  16  )   "
        ]
        
        for expr in expressions_with_whitespace:
            # Should not raise exception
            validate_ast(expr)

    @patch('pyexec_service.logger')
    def test_validate_ast_logging(self, mock_logger):
        """Test that validation logs appropriate messages"""
        # Test valid expression logging
        validate_ast("1 + 2")
        mock_logger.info.assert_called()
        
        # Test invalid expression logging
        with pytest.raises(ValidationError):
            validate_ast("print('hello')")
        mock_logger.warning.assert_called()

    def test_validation_error_properties(self):
        """Test ValidationError custom exception properties"""
        message = "Test error message"
        category = ErrorCategory.SYNTAX_ERROR
        
        error = ValidationError(message, category)
        
        assert str(error) == message
        assert error.category == category

    def test_error_categories_defined(self):
        """Test that all error categories are properly defined"""
        expected_categories = [
            'SYNTAX_ERROR',
            'TIMEOUT',
            'FORBIDDEN_FUNCTION',
            'FORBIDDEN_NODE',
            'TOO_LONG',
            'TOO_COMPLEX',
            'RUNTIME_ERROR'
        ]
        
        for category in expected_categories:
            assert hasattr(ErrorCategory, category)
            assert isinstance(getattr(ErrorCategory, category), str)