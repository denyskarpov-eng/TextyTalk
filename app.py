import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, LanceDB
from langchain.chains import RetrievalQA
import os
from PyPDF2 import PdfReader

# Page title
st.set_page_config(page_title='🦜🔗 TextyTalk')


if "api_key" in st.session_state:
    user_input = st.session_state["api_key"]
else:
    user_input = st.text_input('Enter your API key', type='password', key="api_key_input")
    if user_input:
        st.session_state["api_key"] = user_input

if user_input:
    st.write('<p style="color:green;">API key is being used in the session and will be automatically deleted once the app is closed</p>', unsafe_allow_html=True)
    #st.empty()




def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        file_name = uploaded_file.name
        # Extract text from PDF file
        if uploaded_file.type == 'application/pdf':
            pdf_reader = PdfReader(uploaded_file)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=text)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = Chroma.from_texts(chunks, embeddings)
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            return qa.run(query_text)
        elif os.path.splitext(file_name)[1] == ".docx":
            st.write("File name:", file_name)
            documents = [uploaded_file.read().decode('utf-8')]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            texts = text_splitter.create_documents(documents)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = Chroma.from_documents(texts, embeddings)
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            return qa.run(query_text)
        else:
            # Extract text from TXT file
            documents = [uploaded_file.read().decode()]
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.create_documents(documents)
            # Select embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            # Create a vectorstore from documents
            db = Chroma.from_documents(texts, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()
            # Create QA chain
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            return qa.run(query_text)


# File upload
uploaded_file = st.file_uploader('Upload a document', type=['txt', 'pdf', 'docx'])
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []


if uploaded_file and query_text:
    with st.spinner('Calculating...'):
        response = generate_response(uploaded_file, user_input, query_text)
        result.append(response)
        

if len(result):
    st.info(response)
