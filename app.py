import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Correct import for FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to handle PDF text extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain for QA
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and provide a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Allowing dangerous deserialization (Make sure your file is safe!)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📄", layout="wide")
    st.title("📚 Chat with Multiple PDFs using Gemini 💬")
    st.markdown("Upload your PDFs, ask questions, and get answers based on the document content.")

    # Sidebar for uploading PDFs
    st.sidebar.header("📂 Upload Your PDFs")
    pdf_docs = st.sidebar.file_uploader("Select PDF Files", accept_multiple_files=True, type=["pdf"])

    # Ensure there are files uploaded before continuing
    if pdf_docs and st.sidebar.button("Submit & Process PDFs"):
        with st.spinner("Processing your PDF files..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success("PDFs processed and ready for questions!")

    # Question input area
    st.markdown("💡 **Tip**: Ask something specific about your uploaded PDFs for better answers.")
    user_question = st.text_input("What would you like to ask?", placeholder="Type your question here...")

    # Display the response if a question is asked and PDFs are processed
    if user_question and pdf_docs:
        with st.spinner("Thinking..."):
            response = user_input(user_question)
            if response:
                st.success("Answer retrieved!")
                st.markdown(f"### 💬 Response: \n\n{response}")
            else:
                st.error("Sorry, no relevant information found in the context.")
    elif not pdf_docs:
        st.warning("Please upload PDF files before asking a question.")

    # Footer information
    st.markdown("---")
    st.markdown(
        """
        💬 **How it works**: This app uses **Google's Gemini** (Bard API) to answer questions based on the content of your PDFs.
        Upload the PDFs, ask questions, and the app will retrieve answers from the provided context.
        """
    )

# Run the app
if __name__ == "__main__":
    main()
