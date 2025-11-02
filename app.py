import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
 
# Load FAISS index and metadata
@st.cache_resource
def load_faiss_index_and_metadata():
    index = faiss.read_index("faq_index.faiss")
    with open("faq_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata["questions"], metadata["answers"]
 
# Load embedding and generation models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embed_model, gen_tokenizer, gen_model
 
# RAG-style chatbot function
def rag_chatbot(query, index, questions, answers, embed_model, gen_tokenizer, gen_model, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
 
    retrieved_context = ""
    for idx in indices[0]:
        retrieved_context += f"Q: {questions[idx]}\nA: {answers[idx]}\n\n"
 
    prompt = f"""You are an AI assistant for Aurora Skies Airways.
Use only the following FAQ context to answer the question accurately.
 
Context:
{retrieved_context}
 
Question: {query}
 
Answer:"""
 
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_new_tokens=150)
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
 
# Streamlit UI
st.set_page_config(page_title="Aurora Skies Airways Chatbot", page_icon="✈️")
st.title("✈️ Aurora Skies Airways Chatbot")
st.write("Ask any question related to Aurora Skies Airways. Type your query below:")
 
# Load resources
index, questions, answers = load_faiss_index_and_metadata()
embed_model, gen_tokenizer, gen_model = load_models()
 
# User input
user_query = st.text_input("💬 Your Question", "")
 
if user_query:
    with st.spinner("Generating response..."):
        response = rag_chatbot(user_query, index, questions, answers, embed_model, gen_tokenizer, gen_model)
    st.markdown("### 🤖 Chatbot Response")
    st.write(response)