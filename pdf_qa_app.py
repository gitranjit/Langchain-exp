import streamlit as st
import os
import tempfile
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "ranjit koragoanakr api key"   # 

st.set_page_config(page_title="PDF QA App", layout="centered")
st.title("📄 Ask Questions About Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Process only once
if uploaded_file and "qa_chain" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):
        loader = UnstructuredPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embed = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(chunks, embed)
        retriever = vectordb.as_retriever()

        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.session_state.qa_chain = qa_chain
        st.session_state.chunks = chunks

    st.success("PDF processed. Ready to answer questions!")

# Show chunks
if "chunks" in st.session_state:
    with st.expander("🧩 Show extracted chunks"):
        for i, doc in enumerate(st.session_state.chunks[:3]):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content)

# Ask questions
query = st.text_input("Ask something about the PDF:")
if query and "qa_chain" in st.session_state:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke(query)
    st.markdown("### 💬 Answer:")
    st.write(response)
