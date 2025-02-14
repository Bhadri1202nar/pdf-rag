from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Load environment variables (including your Google Gemini API key)
load_dotenv()

# Streamlit app title
st.title("PDF Q&A")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and process the PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    st.write(f"Total number of document chunks: {len(docs)}")
    
    # Create embeddings and FAISS vector store using Google Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Using Google Gemini embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # Define LLM and prompt using Google Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([ 
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create the retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Input for user question
    user_question = st.text_input("Ask me something about the PDF")
    
    if user_question:
        with st.spinner("Searching..."):
            response = rag_chain.invoke({"input": user_question})
            st.write("### Answer:")
            st.write(response.get("answer", "No answer found."))
    
    # Clean up the file after processing
    os.remove(file_path)
