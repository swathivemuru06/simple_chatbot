from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from .prompt import prompt
# import prompt
# print(dir(prompt))
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from pinecone import ServerlessSpec
load_dotenv()
load_dotenv(".env")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# print("Pinecone:", PINECONE_API_KEY)
# print("Groq:", GROQ_API_KEY)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)
def load_pdf_files(data_path):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, data_path)

    loader = DirectoryLoader(data_dir, glob="*.pdf")
    documents = loader.load()

    return documents
extracted_data = load_pdf_files("data")

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """given a list of documents, return a list of documents with only the page content and source metadata"""
    minimal_docs: List[Document] = []
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content,metadata={"source": src}))      
        
    return minimal_docs
minimal_docs=filter_to_minimal_docs(extracted_data)
def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs
split_docs=text_splitter(minimal_docs)
print(f"Number of documents after splitting: {len(split_docs)}")

def download_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs={"device": "cuda"if torch.cuda.is_available() else "cpu"})
    return embeddings
embedding =download_embeddings()
pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="simple-chatbot"
if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index=pc.Index(index_name)
docsearch = PineconeVectorStore.from_existing_index(embedding=embedding, index_name=index_name)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrived_docs=retriever.invoke("what are embeddings?")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("human", "Context: {context}\n\nQuestion: {input}")
    ]
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)
response=rag_chain.invoke("can you give us  brief summary about chapter 1?")
print(response.content)