import streamlit as st
from langchain_helper import get_pdf_text, get_text_chunks, get_vector_store, user_input

def main():
    # Set the configuration for the Streamlit page
    st.set_page_config(page_title="PDF Summarization Chatbot", page_icon=":robot:", layout="wide")

    # Add custom CSS for better styling, including Google Fonts
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

            .main-header {
                background-color: #4CAF50;
                padding: 10px;
                text-align: center;
                color: white;
                font-size: 36px;
                font-family: 'Lobster', cursive;
            }
            .footer {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                text-align: center;
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
                border-radius: 8px;
            }
            .stButton>button:hover {
                background-color: white;
                color: #4CAF50;
                border: 2px solid #4CAF50;
            }
        </style>
        <div class="main-header">
            Real-Time PDF Summarization Chatbot
        </div>
    """, unsafe_allow_html=True)

    # Section for uploading PDF files
    st.header("Upload PDF Files")
    pdf_docs = st.file_uploader("Upload your PDF files here", accept_multiple_files=True)

    # Initialize session state for raw_text if it doesn't already exist
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""

    # Button to process the uploaded PDFs
    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text  # Store the raw text in session state
                # Chunk the extracted text for processing
                text_chunks = get_text_chunks(raw_text)
                # Create a vector store for the text chunks
                get_vector_store(text_chunks)
                st.success("Processing completed!")
        else:
            st.warning("Please upload at least one PDF file.")

    # Main chat area for interacting with the user
    st.subheader("Ask Questions About Your PDFs")
    user_question = st.text_input("Enter your question here and press Enter")
    if user_question:
        if st.session_state.raw_text:
            with st.spinner("Generating answer..."):
                # Get response based on user input
                response = user_input(user_question)
                st.markdown(f"**User:** {user_question}")
                st.markdown(f"**Bot:** {response}")
        else:
            st.warning("Please upload and process a PDF file first.")

    # Footer section
    st.markdown("""
        <div class="footer">
            © TOUFIQUE HASAN - 2024
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()