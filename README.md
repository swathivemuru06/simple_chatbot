# simple_chatbot
RAG Chatbot Using LangChain, HuggingFace, and Pinecone

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on custom documents such as PDFs and text files. It uses the LangChain ecosystem along with HuggingFace embeddings, Pinecone vector database, and large language models like Groq or OpenAI.

The main goal of this project is to combine document retrieval with language generation so that responses are accurate and based on provided data instead of general knowledge.

Features

The project includes the following features:

- Loading and processing documents such as PDFs and text files  
- Splitting text into smaller chunks for better understanding  
- Generating embeddings using HuggingFace models  
- Storing embeddings in Pinecone vector database  
- Retrieving relevant information using similarity search  
- Generating responses using LLM (Groq )  
- Interactive user interface using Streamlit  
- Secure handling of API keys using environment variables  

Project Structure

The project is organized as follows:

simple_chatbot/

- src/
  - helper.py (core RAG logic)
  - prompt.py (prompt templates)
  - __init__.py  

- research/
  - trials.ipynb (experiments)

- app.py (Streamlit application)
- source_index.py (vector database setup using Pinecone)
- requirements.txt (dependencies)
- .env (API keys)
- setup.py
- README.md

Setup Instructions
Step 1: Clone the repository
git clone <your-repo-url>  
cd simple_chatbot  
Step 2: Install dependencies 
pip install -r requirements.txt  

Step 4: Create a .env file
Add your API keys in a .env file in the root directory:


GROQ_API_KEY=your_groq_api_key  
PINECONE_API_KEY=your_pinecone_api_key  

Running the Application

To run the Streamlit application:

streamlit run app.py  

To test the core functionality:

python src/helper.py  

How the System Works

1. Documents are loaded using loaders such as PyPDFLoader or DirectoryLoader  
2. The text is split into smaller chunks using a text splitter  
3. Each chunk is converted into embeddings using HuggingFace models  
4. The embeddings are stored in Pinecone vector database  
5. When a query is asked, relevant chunks are retrieved using similarity search  
6. The retrieved context is passed to a language model to generate a final answer  

Example Use Case

A user can upload a PDF such as a book or research paper and ask questions like:

- Summarize the document  
- What are the key concepts  
- Explain a specific topic  

The chatbot will respond based on the content of the document.

Technologies Used

- LangChain  
- HuggingFace Transformers  
- Pinecone (Vector Database)  
- Streamlit  
- Groq 
- Python 3.11  


