import os
import json
import warnings
warnings.filterwarnings('ignore')

# Streamlit UI
import streamlit as st

#Loading data
with open('career_data.json') as f:
    items = json.load(f)

# Job Chunks
job_chunks = []
for job in items['jobs']:
    chunk1 = (
        job['title'] + '. ' + job['description'] + '. ' +
        job['experience_level'] + ', Skills: ' + ', '.join(job['skills']) +
        '. ' + job['company'] + '. ' + job['location']
    )
    job_chunks.append(chunk1)

# Mentor Chunk
mentor_chunks = []
for mentor in items['mentors']:
    chunk2 = (
        'Expertise: ' + ', '.join(mentor['expertise']) + '. ' +
        mentor['industry'] + '. ' + mentor['bio'] + '. ' +
        str(mentor['experience_years']) + '. ' + mentor['name']
    )
    mentor_chunks.append(chunk2)

# Event Chunk
event_chunks = []
for event in items['events']:
    chunk3 = (
        event['title'] + '. ' + event['description'] + '. Topics: ' +
        ', '.join(event['topics']) + '. ' + event['date'] + '. ' + event['format']
    )
    event_chunks.append(chunk3)

# Course Chunk
course_chunks = []
for course in items['courses']:
    chunk4 = (
        course['title'] + '. ' + course['description'] + '. ' +
        course['level'] + course['duration'] + '. Skills: ' +
        ', '.join(course['skills']) + '. ' + course['provider']
    )
    course_chunks.append(chunk4)

# Combining the chunks
all_chunks = job_chunks + mentor_chunks + course_chunks + event_chunks

# Creating Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-V2', device='cpu')

# Encoding Chunks
embeddings = model.encode(all_chunks, convert_to_numpy=True)

import faiss
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Persist index and chunks
faiss.write_index(index, 'faiss.idx')
np.save('chunks.npy', np.array(all_chunks, dtype=object))

# Wrapping FAISS index as Retriever in LangChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Reload your embeddings and index
g_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-V2')
db = FAISS.from_texts(all_chunks, embedding=g_embeddings)
db.save_local('faiss_store')
db = FAISS.load_local('faiss_store', g_embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 3})

# Connect Clause as LLM in LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.1,
    max_output_token=256,
    google_api_key="AIzaSyABInyiD2_lBtbrVSTCkoJYE8lhiOlBXqQ"
)

# Building Retrieval QA Chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff'
)

# Chatbot function
def chatbot_qa(user_query: str) -> str:
    res = qa_chain.invoke({'query': user_query})
    return res.get('result') or str(res)

# --- Streamlit Wrapper ---

def main():
    st.set_page_config(page_title="Career RAG Chatbot", layout="centered")
    st.title("💼 Career Data Chatbot")

    query = st.text_input("Ask me anything about your career data:")
    if query:
        with st.spinner("Thinking..."):
            answer = chatbot_qa(query)
        st.markdown("**Answer:**")
        st.write(answer)

if __name__ == "__main__":
    main()
