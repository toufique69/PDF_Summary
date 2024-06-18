# Real-Time PDF Summarization Chatbot

## Overview
The Real-Time PDF Summarization Chatbot is a web application that allows users to upload PDF files and interact with the content in a conversational style, similar to ChatGPT. The application reads and summarizes the content of the uploaded PDFs and provides answers to user queries in real-time.

![1](https://github.com/toufique69/PDF_Summary/assets/13836636/0678da0e-e6df-44e4-a3ce-740d4ce3931c)


## Features
- **PDF Upload**: Users can upload multiple PDF files.
- **Real-Time Summarization**: The application processes and summarizes the content of the PDFs in real-time.
- **Chatbot Interface**: Users can ask questions about the uploaded PDFs and receive detailed answers.
- **Stylish Interface**: Custom CSS for an enhanced user experience.

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- OpenAI API key

### Steps
1. **Clone the repository:**
2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the OpenAI API key:**
    Create a `.env` file in the project root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```
   
4. **Run the application:**
    ```bash
    streamlit run main.py
    ```
   
## Usage
1. Open your web browser and go to the local server URL displayed in the terminal (usually `http://localhost:8501`).
2. Upload your PDF files using the upload section.
3. Once the PDFs are processed, you can ask questions about the content in the chat section.
4. The bot will respond to your questions based on the content of the uploaded PDFs.

## File Descriptions
- **main.py**: The main Streamlit application file that sets up the web interface and handles user interactions.
- **langchain_helper.py**: Contains functions for extracting text from PDFs, creating text chunks, generating embeddings, and handling user queries.
- **unit_test.py**: Unit tests for the functions in `langchain_helper.py`.

## Technologies Used
- **Streamlit**: For building the web interface.
- **LangChain**: For handling text splitting, embeddings, and question-answering chains.
- **OpenAI API**: For generating embeddings and chat responses.
- **PyPDF2**: For extracting text from PDF files.
- **FAISS**: For creating and handling vector stores.
- **unittest**: For unit testing the application.


© TOUFIQUE HASAN - 2024
