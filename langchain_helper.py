import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import openai

# Load environment variables from a .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Function to split text into chunks for better processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split text into chunks
    return chunks

# Function to create and save vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Initialize embeddings model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create vector store from text chunks
    vector_store.save_local("faiss_index")  # Save vector store locally

# Function to create a conversational chain using a prompt template
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """  # Define prompt template for the QA model
    model = ChatOpenAI(model="gpt-4", temperature=0.3)  # Initialize the chat model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Create prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load QA chain with the model and prompt
    return chain

# Function to handle user input and generate a response
def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Initialize embeddings model
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load local vector store
    docs = new_db.similarity_search(user_question)  # Perform similarity search on the vector store
    chain = get_conversational_chain()  # Get conversational chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )  # Generate response using the chain
    return response["output_text"]  # Return the generated response