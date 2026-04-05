import streamlit as st
from source_index import docsearch
from src.helper import llm, format_docs
from src.prompt import prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Chatbot")
st.title("Chatbot with RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Added: trim history to 5000 tokens (approx 4 chars = 1 token)
def trim_history(history, max_tokens=5000):
    while history:
        total_chars = sum(len(m["user"]) + len(m["bot"]) for m in history)
        if total_chars <= max_tokens * 4:
            break
        history.pop(0)  # remove oldest message
    return history

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

for message in st.session_state.chat_history:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**Answer:** {message['bot']}")
    st.markdown("---")

query = st.text_input("Ask something:")

if query:
    response = rag_chain.invoke(query)
    answer = response.content

    st.session_state.chat_history.append({"user": query, "bot": answer})
    st.session_state.chat_history = trim_history(st.session_state.chat_history)  # Added

    st.markdown("---")
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Answer:** {answer}")