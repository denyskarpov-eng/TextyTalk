import streamlit as st
#from docx import Document
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

import os
import io
import tempfile



# Page title
st.set_page_config(page_title='🗣️📃 TextyTalk')


if "api_key" in st.session_state:
    openai_api_key = st.session_state["api_key"]
else:
    openai_api_key = st.text_input('Enter your API key', type='password', key="api_key_input")
    if openai_api_key:
        st.session_state["api_key"] = openai_api_key
        st.write('<p style="color:green;">API key is being used in the session and will be automatically deleted once the app is closed</p>', unsafe_allow_html=True)
        #st.empty()




def generate_embeddings(openai_api_key, uploaded_file):
    global current_db
    # Load document if file is uploaded
    if uploaded_file is not None and current_db is None:
        file_name = uploaded_file.name
        # Extract text from PDF file
        if uploaded_file.type == "application/pdf":

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            loader = PyPDFLoader(temp_file_path)
            
        # Extract text from Microsoft Word file
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = Docx2txtLoader(temp_file_path)     

        # Extract text from Txt file
        elif uploaded_file.type == 'text/plain':
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = TextLoader(temp_file_path)

        else:
            raise ValueError("Unsupported file format. Please upload 'txt', 'pdf' or 'docx' file.")
            
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        docs = splitter.split_documents(pages)
        
        try:
            embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
            db = Chroma.from_documents(docs, embeddings)
            current_db = db
            return db
        except openai.error.OpenAIError as e:
            st.write(e.error["message"])
        
        
 

def generate_response(openai_api_key, query_text):
    global current_db
    if current_db is not None:
        retriever = current_db.as_retriever(search_kwargs={"k": 5})
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)
    else:
        return "No document uploaded yet."

        
# Form input and query
result = []


# File upload
uploaded_file = st.file_uploader('Upload a document', type=['txt', 'pdf', 'docx'])
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)




current_db = None
 


if  openai_api_key is not None and current_db is None and uploaded_file is not None:
    st.write("Generating embeddings")
    current_db = generate_embeddings(openai_api_key, uploaded_file)
    


if current_db and query_text:
    with st.spinner('Calculating...'):
        response = generate_response(openai_api_key, query_text)
        result.append(response)


if len(result):
    st.info(response)
