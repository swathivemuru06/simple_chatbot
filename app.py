import streamlit as st
from source_index import docsearch
from src.helper import llm, format_docs
from src.prompt import prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Chatbot")
st.title("Chatbot with RAG")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("human", "Context: {context}\n\nQuestion: {input}")
])

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt_template
    | llm
)

query = st.text_input("Ask something:")

if query:
    response = rag_chain.invoke(query)
    answer = response.content

    st.markdown("---")
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Answer:** {answer}")