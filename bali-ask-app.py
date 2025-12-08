import streamlit as st
import os
import requests
import fitz
from io import BytesIO

from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2

# ---- PDF EXTRACT FUNCTION ----
def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_bytes = BytesIO(response.content)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---- API KEYS ----
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ---- LOAD & PROCESS PDF ----
@st.cache_resource
def load_vector_db():
    pdf_text = extract_text_from_pdf(
        "https://raw.githubusercontent.com/Dhihram/RAG_book_sample/main/baliguide-sample-pdf.pdf"
    )

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(pdf_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    index = IndexFlatL2(
        len(embedding_model.embed_query("hello world"))
    )

    vector_db = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_db.add_documents(docs)
    return vector_db


vector_db = load_vector_db()

# ---- LLM ----
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ---- UI ----
st.title("Bali Travel RAG Assistant")

question = st.text_input("Ask a question about Bali:")

if question:
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using only this context:

    {context}

    Question:
    {question}
    """

    answer = llm.invoke(prompt)

    st.write("### ✅ Answer")
    st.write(answer.content)