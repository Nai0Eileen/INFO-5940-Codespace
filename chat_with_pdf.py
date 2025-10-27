import streamlit as st
import os
from openai import OpenAI
from os import environ
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

client = OpenAI(
	api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)

st.title("📝 File Q&A with OpenAI")
uploaded_files = st.file_uploader("Upload an article", type=("txt", "md","pdf"), accept_multiple_files=True)

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_files:
    # Read the content of the uploaded file
    file_content = ""

    for uploaded_file in uploaded_files:
        file_content += f"\n\n[Source: {uploaded_file.name}]\n"
        if uploaded_file.name.lower().endswith(".pdf"):  #so the model can read pdf
            reader = PdfReader(uploaded_file)
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    file_content += text
        else:
            file_content += uploaded_file.read().decode("utf-8")
    print(file_content)

    #RAG chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_content)
    
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=os.environ["API_KEY"],
        base_url="https://api.ai.it.cornell.edu",
    )

    vectordb = Chroma.from_texts(chunks, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    top_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([d.page_content for d in top_docs])

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="openai.gpt-4o",  # Change this to a valid model name
            messages=[
                {"role": "system", "content": f"Here's the content of the file:\n\n{context}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})