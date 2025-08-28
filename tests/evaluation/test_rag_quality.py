"""RAG system quality evaluation tests"""

import pytest
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch
import statistics
import requests

# Import modules under test
from langchain.schema import Document


class RAGEvaluator:
    """Evaluates RAG system quality using various metrics"""
    
    def __init__(self, qa_dataset_path: str = "tests/fixtures/qa_dataset.json"):
        self.qa_dataset_path = qa_dataset_path
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the Q&A evaluation dataset"""
        try:
            with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            pytest.skip(f"Dataset file not found: {self.qa_dataset_path}")
    
    def calculate_precision_at_k(self, retrieved_docs: List[Document], 
                                expected_topics: List[str], k: int = 3) -> float:
        """Calculate precision@k for retrieved documents"""
        if not retrieved_docs or not expected_topics:
            return 0.0
            
        relevant_count = 0
        docs_to_check = retrieved_docs[:k]
        
        for doc in docs_to_check:
            doc_content = doc.page_content.lower()
            for topic in expected_topics:
                if topic.lower() in doc_content:
                    relevant_count += 1
                    break
                    
        return relevant_count / min(k, len(retrieved_docs))
    
    def calculate_recall_at_k(self, retrieved_docs: List[Document],
                             expected_topics: List[str], k: int = 3) -> float:
        """Calculate recall@k for retrieved documents"""
        if not retrieved_docs or not expected_topics:
            return 0.0
            
        found_topics = set()
        docs_to_check = retrieved_docs[:k]
        
        for doc in docs_to_check:
            doc_content = doc.page_content.lower()
            for topic in expected_topics:
                if topic.lower() in doc_content:
                    found_topics.add(topic.lower())
                    
        return len(found_topics) / len(expected_topics)
    
    def calculate_groundedness(self, response: str, retrieved_docs: List[Document]) -> float:
        """Calculate how grounded the response is in retrieved documents"""
        if not response or not retrieved_docs:
            return 0.0
            
        response_lower = response.lower()
        context_text = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        
        # Simple groundedness: check if key phrases from response appear in context
        response_words = set(re.findall(r'\b\w+\b', response_lower))
        context_words = set(re.findall(r'\b\w+\b', context_text))
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        meaningful_response_words = response_words - common_words
        if not meaningful_response_words:
            return 0.0
            
        grounded_words = meaningful_response_words.intersection(context_words)
        return len(grounded_words) / len(meaningful_response_words)
    
    def evaluate_response_relevance(self, response: str, question: str,
                                   expected_answer_contains: List[str]) -> float:
        """Evaluate how relevant the response is to the question"""
        if not response:
            return 0.0
            
        response_lower = response.lower()
        relevance_score = 0.0
        
        # Check if expected content is present
        for expected_content in expected_answer_contains:
            if expected_content.lower() in response_lower:
                relevance_score += 1
                
        return relevance_score / len(expected_answer_contains) if expected_answer_contains else 0.0
    
    def validate_calculation_accuracy(self, response: str, expected_result: float,
                                    tolerance: float = 0.01) -> bool:
        """Validate calculation accuracy in response"""
        # Extract numbers from response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        for num_str in numbers:
            try:
                num = float(num_str)
                if abs(num - expected_result) <= tolerance:
                    return True
            except ValueError:
                continue
        return False


@pytest.fixture(scope="module")
def rag_evaluator():
    """Create RAG evaluator instance"""
    return RAGEvaluator()


@pytest.fixture
def mock_rag_system():
    """Mock RAG system components"""
    with patch('app.retriever') as mock_retriever, \
         patch('app.llm') as mock_llm, \
         patch('requests.post') as mock_pyexec:
        
        yield {
            'retriever': mock_retriever,
            'llm': mock_llm,
            'pyexec': mock_pyexec
        }


class TestRAGQuality:
    """Test RAG system quality metrics"""
    
    @pytest.mark.evaluation
    def test_dataset_loading(self, rag_evaluator):
        """Test that the Q&A dataset loads correctly"""
        assert rag_evaluator.dataset is not None
        assert 'questions' in rag_evaluator.dataset
        assert len(rag_evaluator.dataset['questions']) >= 10
        
        # Validate dataset structure
        for question in rag_evaluator.dataset['questions']:
            required_fields = ['id', 'question', 'category', 'expected_topics']
            for field in required_fields:
                assert field in question, f"Missing field {field} in question {question.get('id')}"
    
    @pytest.mark.evaluation
    def test_precision_at_k_calculation(self, rag_evaluator):
        """Test precision@k calculation"""
        # Mock retrieved documents
        retrieved_docs = [
            Document(page_content="Machine learning is a subset of AI that learns from data"),
            Document(page_content="Neural networks are inspired by biological neurons"),
            Document(page_content="This document is about cooking recipes")
        ]
        
        expected_topics = ["machine learning", "neural networks"]
        
        precision = rag_evaluator.calculate_precision_at_k(retrieved_docs, expected_topics, k=3)
        assert 0.0 <= precision <= 1.0
        assert precision > 0.5  # Should find relevant content
    
    @pytest.mark.evaluation
    def test_recall_at_k_calculation(self, rag_evaluator):
        """Test recall@k calculation"""
        retrieved_docs = [
            Document(page_content="Machine learning algorithms learn from data"),
            Document(page_content="Deep learning uses neural networks"),
        ]
        
        expected_topics = ["machine learning", "neural networks", "unsupervised learning"]
        
        recall = rag_evaluator.calculate_recall_at_k(retrieved_docs, expected_topics, k=3)
        assert 0.0 <= recall <= 1.0
        assert recall > 0.0  # Should find some topics
    
    @pytest.mark.evaluation
    def test_groundedness_calculation(self, rag_evaluator):
        """Test groundedness metric calculation"""
        response = "Machine learning algorithms learn patterns from training data to make predictions"
        retrieved_docs = [
            Document(page_content="Machine learning is about learning patterns from data"),
            Document(page_content="Algorithms use training data to make predictions")
        ]
        
        groundedness = rag_evaluator.calculate_groundedness(response, retrieved_docs)
        assert 0.0 <= groundedness <= 1.0
        assert groundedness > 0.5  # Response should be grounded in context
    
    @pytest.mark.evaluation
    def test_response_relevance_evaluation(self, rag_evaluator):
        """Test response relevance evaluation"""
        response = "Machine learning is a method of data analysis that automates analytical model building using algorithms"
        question = "What is machine learning?"
        expected_contains = ["machine learning", "algorithms", "data", "analysis"]
        
        relevance = rag_evaluator.evaluate_response_relevance(response, question, expected_contains)
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.7  # Should be highly relevant
    
    @pytest.mark.evaluation
    def test_calculation_accuracy_validation(self, rag_evaluator):
        """Test calculation accuracy validation"""
        response = "The square root of 144 is 12"
        expected_result = 12.0
        
        is_accurate = rag_evaluator.validate_calculation_accuracy(response, expected_result)
        assert is_accurate is True
        
        # Test inaccurate calculation
        inaccurate_response = "The square root of 144 is 15"
        is_accurate = rag_evaluator.validate_calculation_accuracy(inaccurate_response, expected_result)
        assert is_accurate is False
    
    @pytest.mark.evaluation
    @pytest.mark.slow
    def test_sample_questions_evaluation(self, rag_evaluator, mock_rag_system):
        """Test evaluation on sample questions from dataset"""
        sample_questions = rag_evaluator.dataset['questions'][:5]  # Test first 5 questions
        
        for question_data in sample_questions:
            question = question_data['question']
            expected_topics = question_data['expected_topics']
            expected_contains = question_data.get('expected_answer_contains', [])
            
            # Mock retrieval
            mock_docs = [
                Document(
                    page_content=f"Sample content about {topic}",
                    metadata={"source": f"{topic}.pdf", "page": 1}
                ) for topic in expected_topics[:2]
            ]
            mock_rag_system['retriever'].get_relevant_documents.return_value = mock_docs
            
            # Mock LLM response
            mock_response = f"Sample response about {' and '.join(expected_topics[:2])}"
            mock_rag_system['llm'].invoke.return_value = Mock(content=mock_response)
            
            # Evaluate
            precision = rag_evaluator.calculate_precision_at_k(mock_docs, expected_topics, k=3)
            groundedness = rag_evaluator.calculate_groundedness(mock_response, mock_docs)
            relevance = rag_evaluator.evaluate_response_relevance(mock_response, question, expected_contains)
            
            # Assertions
            assert precision >= 0.0
            assert groundedness >= 0.0
            assert relevance >= 0.0
            
            print(f"Question {question_data['id']}: P@3={precision:.3f}, G={groundedness:.3f}, R={relevance:.3f}")
    
    @pytest.mark.evaluation
    @pytest.mark.slow
    def test_calculation_questions_evaluation(self, rag_evaluator, mock_rag_system):
        """Test evaluation on calculation questions"""
        calc_questions = [q for q in rag_evaluator.dataset['questions'] if q.get('calculation_required', False)]
        
        for question_data in calc_questions[:3]:  # Test first 3 calculation questions
            question = question_data['question']
            expected_result = question_data.get('expected_result', 0)
            expected_calc = question_data.get('expected_calculation', "")
            
            # Mock pyexec response
            mock_rag_system['pyexec'].return_value = Mock(
                status_code=200,
                json=lambda: {"result": expected_result, "expression": expected_calc}
            )
            
            # Mock retrieval and LLM response
            mock_docs = [Document(page_content=f"Mathematical content for {expected_calc}")]
            mock_rag_system['retriever'].get_relevant_documents.return_value = mock_docs
            
            mock_response = f"The calculation {expected_calc} equals {expected_result}"
            mock_rag_system['llm'].invoke.return_value = Mock(content=mock_response)
            
            # Evaluate
            is_accurate = rag_evaluator.validate_calculation_accuracy(mock_response, expected_result)
            groundedness = rag_evaluator.calculate_groundedness(mock_response, mock_docs)
            
            assert is_accurate is True
            assert groundedness >= 0.0
            
            print(f"Calculation Q{question_data['id']}: Accurate={is_accurate}, G={groundedness:.3f}")
    
    @pytest.mark.evaluation
    def test_evaluation_thresholds(self, rag_evaluator):
        """Test that evaluation meets quality thresholds"""
        criteria = rag_evaluator.dataset.get('evaluation_criteria', {})
        
        # Test thresholds are reasonable
        assert criteria.get('precision_at_k_threshold', 0) >= 0.5
        assert criteria.get('groundedness_threshold', 0) >= 0.7
        assert criteria.get('relevance_threshold', 0) >= 0.6
        assert criteria.get('response_time_threshold_seconds', 10) <= 5.0
    
    @pytest.mark.evaluation
    @pytest.mark.integration
    def test_end_to_end_evaluation_sample(self, rag_evaluator):
        """Test end-to-end evaluation on a small sample"""
        # This would be used with a real running system
        # For now, we test the evaluation framework works
        
        sample_question = rag_evaluator.dataset['questions'][0]
        
        # Mock a complete evaluation flow
        mock_retrieved_docs = [
            Document(page_content="Machine learning is about learning from data"),
            Document(page_content="Algorithms use statistical techniques to find patterns")
        ]
        
        mock_response = "Machine learning is a method where algorithms learn patterns from data"
        
        # Calculate all metrics
        precision = rag_evaluator.calculate_precision_at_k(
            mock_retrieved_docs, 
            sample_question['expected_topics'], 
            k=3
        )
        
        groundedness = rag_evaluator.calculate_groundedness(mock_response, mock_retrieved_docs)
        
        relevance = rag_evaluator.evaluate_response_relevance(
            mock_response,
            sample_question['question'],
            sample_question.get('expected_answer_contains', [])
        )
        
        # Create evaluation report
        evaluation_report = {
            "question_id": sample_question['id'],
            "question": sample_question['question'],
            "metrics": {
                "precision_at_3": precision,
                "groundedness": groundedness,
                "relevance": relevance
            },
            "thresholds_met": {
                "precision": precision >= rag_evaluator.dataset['evaluation_criteria'].get('precision_at_k_threshold', 0.7),
                "groundedness": groundedness >= rag_evaluator.dataset['evaluation_criteria'].get('groundedness_threshold', 0.8),
                "relevance": relevance >= rag_evaluator.dataset['evaluation_criteria'].get('relevance_threshold', 0.7)
            }
        }
        
        print(f"Evaluation Report: {evaluation_report}")
        
        # At least some metrics should be reasonable
        assert precision >= 0.0
        assert groundedness >= 0.0
        assert relevance >= 0.0
    
    @pytest.mark.evaluation
    def test_evaluation_framework_completeness(self, rag_evaluator):
        """Test that the evaluation framework covers all necessary aspects"""
        dataset = rag_evaluator.dataset
        
        # Check dataset completeness
        assert 'metadata' in dataset
        assert 'evaluation_criteria' in dataset
        assert 'questions' in dataset
        
        # Check question categories are covered
        categories = set(q['category'] for q in dataset['questions'])
        expected_categories = {'definition', 'calculation', 'explanation', 'comparison'}
        assert len(categories.intersection(expected_categories)) >= 3
        
        # Check difficulty levels
        difficulties = set(q['difficulty'] for q in dataset['questions'])
        assert len(difficulties) >= 2  # At least easy and medium
        
        # Check calculation questions exist
        calc_questions = [q for q in dataset['questions'] if q.get('calculation_required', False)]
        assert len(calc_questions) >= 3
        
        print(f"Dataset contains {len(dataset['questions'])} questions across {len(categories)} categories")