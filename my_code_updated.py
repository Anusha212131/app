import os
import json
import warnings
import logging
from datetime import datetime
from functools import lru_cache

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
import uvicorn

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Utility: sliding-window text chunking
def chunk_text(text, chunk_size=500, overlap=50):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        chunk = tokens[start : start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

# Load initial data
with open('career_data.json', 'r') as f:
    items = json.load(f)

# Build documents with metadata
docs = []
for idx, job in enumerate(items.get('jobs', [])):
    text = f"{job['title']}. {job['description']}. {job['experience_level']}. Skills: {', '.join(job['skills'])}."
    for chunk in chunk_text(text):
        docs.append(Document(page_content=chunk, metadata={'source': 'job', 'id': idx}))
for idx, mentor in enumerate(items.get('mentors', [])):
    text = f"Expertise: {', '.join(mentor['expertise'])}. {mentor['industry']}. {mentor['bio']}"
    for chunk in chunk_text(text):
        docs.append(Document(page_content=chunk, metadata={'source': 'mentor', 'id': idx}))
for idx, event in enumerate(items.get('events', [])):
    text = f"{event['title']}. {event['description']}. Topics: {', '.join(event['topics'])}."
    for chunk in chunk_text(text):
        docs.append(Document(page_content=chunk, metadata={'source': 'event', 'id': idx}))
for idx, course in enumerate(items.get('courses', [])):
    text = f"{course['title']}. {course['description']}. Level: {course['level']}. Skills: {', '.join(course['skills'])}."
    for chunk in chunk_text(text):
        docs.append(Document(page_content=chunk, metadata={'source': 'course', 'id': idx}))

# Initialize embedding and vector store
embedding_model = SentenceTransformer('all-MiniLM-L6-V2', device='cpu')
hf_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-V2')
vectorstore = LCFAISS.from_documents(docs, embedding=hf_embeddings)
vectorstore.save_local('faiss_store')
vectorstore = LCFAISS.load_local('faiss_store', hf_embeddings, allow_dangerous_deserialization=True)

# Retriever with MMR support
def get_retriever(k: int, fetch_k: int, lambda_mult: float):
    return vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': k, 'fetch_k': fetch_k, 'lambda_mult': lambda_mult},
    )

# Initialize LLM
def get_llm(temperature: float, max_tokens: int):
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=temperature,
        max_output_token=max_tokens,
        google_api_key='AIzaSyABInyiD2_lBtbrVSTCkoJYE8lhiOlBXqQ',
        streaming=True,
    )

# Build conversational RAG chain
def get_chain(llm, retriever):
    system_template = "You are an expert career advisor. Think step by step before answering."
    sys_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{question}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    prompt = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt},
        return_source_documents=True,
    )

# Cache identical queries
def make_cache_key(query, temp, max_tokens, k, fetch_k, lambda_mult):
    return f"{query}|{temp}|{max_tokens}|{k}|{fetch_k}|{lambda_mult}"

@lru_cache(maxsize=128)
def cached_chat(query: str, temp: float, max_tokens: int, k: int, fetch_k: int, lambda_mult: float):
    llm = get_llm(temp, max_tokens)
    retriever = get_retriever(k, fetch_k, lambda_mult)
    chain = get_chain(llm, retriever)
    result = chain({'question': query})
    return result

# Streamlit UI setup
st.set_page_config(page_title='💬 Career Chatbot', layout='wide')
st.title('💬 Career Chatbot')

# Sidebar: controls & file upload
with st.sidebar:
    st.header('Settings')
    temperature = st.slider('Temperature', 0.0, 1.0, 0.1)
    max_tokens = st.slider('Max tokens', 50, 1024, 256)
    k = st.number_input('Retriever k', min_value=1, max_value=10, value=3)
    fetch_k = st.number_input('Retriever fetch_k', min_value=1, max_value=50, value=10)
    lambda_mult = st.slider('MMR diversity λ', 0.0, 1.0, 0.5)
    threshold = st.slider('Fallback threshold (unused)', 0.0, 1.0, 0.2)

    st.markdown('---')
    st.markdown('## Upload & Reindex')
    uploaded = st.file_uploader('Upload JSON/data file', type=['json', 'txt'])
    if st.button('Reindex Uploaded') and uploaded:
        try:
            data = json.load(uploaded) if uploaded.type == 'application/json' else {'text': [uploaded.read().decode()]}
            new_docs = []
            for txt in data.get('text', []):
                for chunk in chunk_text(txt):
                    new_docs.append(Document(page_content=chunk, metadata={'source': 'uploaded'}))
            if new_docs:
                vectorstore.add_documents(new_docs)
                vectorstore.save_local('faiss_store')
                st.success('Reindexed uploaded content.')
        except Exception as e:
            st.error(f'Failed to reindex: {e}')

# Chat interface
if 'history' not in st.session_state:
    st.session_state.history = []
user_input = st.text_input('Your question:', key='input')
if user_input:
    # Retrieve docs
    retr = get_retriever(k, fetch_k, lambda_mult)
    docs = retr.get_relevant_documents(user_input)
    if not docs:
        answer = 'I’m not certain—could you rephrase?'
    else:
        result = cached_chat(user_input, temperature, max_tokens, k, fetch_k, lambda_mult)
        answer = result.get('answer') or result.get('result', '')
        # Attach citations
        cites = {d.metadata.get('source') for d in docs[:k]}
        answer += '\n\n**Sources:** ' + ', '.join(cites)
    # Log interaction
    logging.info(f"Query: {user_input} | Answer: {answer}")
    # Update chat history
    st.session_state.history.append(('You', user_input))
    st.session_state.history.append(('Bot', answer))

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == 'You':
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")

# FastAPI app for async endpoint
api_app = FastAPI()

@api_app.post('/chat')
async def chat_endpoint(payload: dict):
    q = payload.get('question', '')
    res = cached_chat(q, temperature, max_tokens, k, fetch_k, lambda_mult)
    return {'answer': res.get('answer') or res.get('result')}

if __name__ == '__main__':
    uvicorn.run('app:api_app', host='0.0.0.0', port=8000)
