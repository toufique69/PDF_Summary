import unittest
from unittest.mock import patch, MagicMock
import os
from langchain_helper import get_pdf_text, get_text_chunks, get_vector_store, user_input, get_conversational_chain

class TestLangchainHelper(unittest.TestCase):

    @patch('langchain_helper.PdfReader')
    def test_get_pdf_text(self, MockPdfReader):
        # Mock PDF Reader and pages
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = "This is the first page."
        mock_pdf.pages[1].extract_text.return_value = "This is the second page."
        MockPdfReader.return_value = mock_pdf

        pdf_docs = ["dummy.pdf"]
        text = get_pdf_text(pdf_docs)
        expected_text = "This is the first page.This is the second page."
        self.assertEqual(text, expected_text)

    def test_get_text_chunks(self):
        text = "This is a test text. " * 1000  # Create a long string
        chunks = get_text_chunks(text)
        # Check if chunks are created and their length
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 11000)  # Chunk size + overlap

    @patch('langchain_helper.FAISS')
    @patch('langchain_helper.OpenAIEmbeddings')
    def test_get_vector_store(self, MockOpenAIEmbeddings, MockFAISS):
        # Mock embeddings and FAISS
        mock_embeddings = MagicMock()
        MockOpenAIEmbeddings.return_value = mock_embeddings
        mock_faiss = MagicMock()
        MockFAISS.from_texts.return_value = mock_faiss

        text_chunks = ["chunk1", "chunk2"]
        get_vector_store(text_chunks)
        # Check if FAISS vector store is created and saved
        MockFAISS.from_texts.assert_called_once_with(text_chunks, embedding=mock_embeddings)
        mock_faiss.save_local.assert_called_once_with("faiss_index")

    def test_get_conversational_chain(self):
        chain = get_conversational_chain()
        # Check if chain is created
        self.assertIsNotNone(chain)

    @patch('langchain_helper.FAISS')
    @patch('langchain_helper.OpenAIEmbeddings')
    @patch('langchain_helper.get_conversational_chain')
    def test_user_input(self, MockGetConversationalChain, MockOpenAIEmbeddings, MockFAISS):
        # Mock embeddings, FAISS, and conversational chain
        mock_embeddings = MagicMock()
        MockOpenAIEmbeddings.return_value = mock_embeddings
        mock_faiss = MagicMock()
        MockFAISS.load_local.return_value = mock_faiss
        mock_faiss.similarity_search.return_value = ["doc1", "doc2"]
        mock_chain = MagicMock()
        mock_chain.return_value = {"output_text": "This is a response."}
        MockGetConversationalChain.return_value = mock_chain

        user_question = "What is the content?"
        response = user_input(user_question)
        # Check if response is generated
        self.assertEqual(response, "This is a response.")
        mock_faiss.similarity_search.assert_called_once_with(user_question)
        mock_chain.assert_called_once()

if __name__ == "__main__":
    unittest.main()
