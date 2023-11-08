import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores.document import Document as LC_Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os


# Page title
st.set_page_config(page_title='🦜🔗 TextyTalk')


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
        if uploaded_file.type == 'application/pdf':
            pdf_reader = PdfReader(uploaded_file)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=text)
            # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            # db = Chroma.from_texts(chunks, embeddings)
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            st.write(file_name)
            doc = Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text
                print(para)
            text_splitter = CharacterTextSplitter(separator = "\n\n", chunk_size = 1000, chunk_overlap  = 200, length_function = len, is_separator_regex = False,)
            chunks = text_splitter.split_text(text=text)
            # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            # db = Chroma.from_texts(chunks, embeddings)
        else:
            # Extract text from TXT file
            documents = [uploaded_file.read().decode()]
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.create_documents(documents)
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(documents, embedding_function)
            # Create a vectorstore from documents
            db = Chroma.from_documents(texts, embeddings)
        # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # db = Chroma.from_documents(documents, embedding_function)
        current_db = db
        
        return db
 

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


# Initialize the Chroma database variable
@st.cache_data
def initialize_db():
    return None

current_db = initialize_db()
 


if  openai_api_key is not None and current_db is None and uploaded_file is not None:
    st.write("Generating embeddings")
    current_db = generate_embeddings(openai_api_key, uploaded_file)
    


if current_db and query_text:
    with st.spinner('Calculating...'):
        response = generate_response(openai_api_key, query_text)
        result.append(response)

if current_db is not None:
    clear_db = st.button('Clear up database from current files')
    if clear_db:
        current_db = None
        result = []



if len(result):
    st.info(response)
