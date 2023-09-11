import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, LanceDB
from langchain.chains import RetrievalQA
import os

# Page title
st.set_page_config(page_title='🦜🔗 TextyTalk')


if "api_key" in st.session_state:
    user_input = st.session_state["api_key"]
else:
    user_input = st.text_input('Enter your API key', type='password', key="api_key_input")
    if user_input:
        st.session_state["api_key"] = user_input
        print("There")

if user_input:
    st.write('<p style="color:green;">API key is being used in the session and will be automatically deleted once the app is closed</p>', unsafe_allow_html=True)
    #st.empty()

if user_input:
    print("YO Ommm")
else:
    print("NO Noooom")


def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Extract text from PDF file
        if uploaded_file.type == 'application/pdf':
            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
            documents = [pdf_reader.getPage(i).extractText() for i in range(pdf_reader.getNumPages())]
        else:
            documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
uploaded_file = st.file_uploader('Upload a document', type=['txt', 'pdf', 'docx', 'html', 'htm', 'rtf', 'csv', 'xml', 'json'])
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []

container = st.empty()

if uploaded_file and query_text:
    with st.spinner('Calculating...'):
        response = generate_response(uploaded_file, user_input, query_text)
        result.append(response)
        container.write(response)

if len(result):
    st.info(response)



print("USER YOU", user_input)