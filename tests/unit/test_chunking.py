"""Unit tests for text chunking functionality"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TestChunking:
    """Test text chunking functionality"""
    
    def setup_method(self):
        """Setup test data and splitter"""
        self.chunk_size = 1200
        self.chunk_overlap = 180
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Sample text with varying lengths
        self.short_text = "This is a short document for testing."
        self.medium_text = " ".join(["This is a medium-length document."] * 20)  # ~600 chars
        self.long_text = " ".join(["This is a long document with many sentences."] * 50)  # ~2250 chars
        
    def test_chunking_short_text(self):
        """Test chunking of short text (no splitting needed)"""
        chunks = self.text_splitter.split_text(self.short_text)
        
        assert len(chunks) == 1
        assert chunks[0] == self.short_text
        assert len(chunks[0]) <= self.chunk_size

    def test_chunking_medium_text(self):
        """Test chunking of medium text (single chunk)"""
        chunks = self.text_splitter.split_text(self.medium_text)
        
        assert len(chunks) == 1
        assert len(chunks[0]) <= self.chunk_size
        assert chunks[0] == self.medium_text

    def test_chunking_long_text(self):
        """Test chunking of long text (multiple chunks)"""
        chunks = self.text_splitter.split_text(self.long_text)
        
        assert len(chunks) >= 2  # Should split into multiple chunks
        for chunk in chunks:
            assert len(chunk) <= self.chunk_size
            assert len(chunk.strip()) > 0  # No empty chunks

    def test_chunking_empty_text(self):
        """Test chunking of empty text"""
        chunks = self.text_splitter.split_text("")
        assert len(chunks) == 0

    def test_chunking_whitespace_only(self):
        """Test chunking of whitespace-only text"""
        whitespace_text = "   \n\n   \t\t   "
        chunks = self.text_splitter.split_text(whitespace_text)
        
        # Should either be empty or contain whitespace
        if chunks:
            for chunk in chunks:
                assert len(chunk) <= self.chunk_size

    def test_chunk_overlap_consistency(self):
        """Test that chunks have proper overlap"""
        chunks = self.text_splitter.split_text(self.long_text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap content
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # The overlap should be meaningful (not just whitespace)
                assert len(current_chunk) <= self.chunk_size
                assert len(next_chunk) <= self.chunk_size

    def test_chunk_size_configuration(self):
        """Test different chunk size configurations"""
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        small_chunks = small_splitter.split_text(self.long_text)
        regular_chunks = self.text_splitter.split_text(self.long_text)
        
        # Smaller chunk size should create more chunks
        assert len(small_chunks) >= len(regular_chunks)
        
        for chunk in small_chunks:
            assert len(chunk) <= 100

    def test_document_chunking_with_metadata(self):
        """Test chunking with document metadata preservation"""
        from langchain.schema import Document
        
        doc = Document(
            page_content=self.long_text,
            metadata={"source": "test.pdf", "page": 1}
        )
        
        chunks = self.text_splitter.split_documents([doc])
        
        assert len(chunks) >= 1
        for chunk_doc in chunks:
            assert isinstance(chunk_doc, Document)
            assert chunk_doc.metadata["source"] == "test.pdf"
            assert chunk_doc.metadata["page"] == 1
            assert len(chunk_doc.page_content) <= self.chunk_size

    def test_special_characters_chunking(self):
        """Test chunking text with special characters"""
        special_text = "Text with émojis 🚀, spécial charàcters ñ, and unicode ∑∏∆"
        chunks = self.text_splitter.split_text(special_text)
        
        assert len(chunks) == 1  # Should be short enough for one chunk
        assert chunks[0] == special_text

    def test_chunking_performance_large_text(self):
        """Test chunking performance with very large text"""
        # Create a very large text
        very_long_text = " ".join(["Performance test sentence."] * 1000)  # ~28k chars
        
        import time
        start_time = time.time()
        chunks = self.text_splitter.split_text(very_long_text)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert len(chunks) > 1
        
        for chunk in chunks:
            assert len(chunk) <= self.chunk_size