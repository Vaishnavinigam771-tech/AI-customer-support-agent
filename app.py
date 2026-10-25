import streamlit as st

from src.document_processor import (
    load_documents,
    split_documents
)

from src.vector_store import (
    create_vector_store
)

from src.graph import build_graph

from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="ShopEase AI Support",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 ShopEase AI Customer Support Agent")

st.caption(
    "Agentic AI + RAG + LangGraph + Groq"
)


# Initialize vector database
if "vector_ready" not in st.session_state:

    with st.spinner(
        "Preparing knowledge base..."
    ):

        documents = load_documents()

        chunks = split_documents(
            documents
        )

        create_vector_store(
            chunks
        )

        st.session_state.vector_ready = True


# Initialize graph
if "graph" not in st.session_state:

    st.session_state.graph = build_graph()


# Chat history
if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# User input
question = st.chat_input(
    "Ask about orders, refunds, shipping or payments..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            state = {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ],

                "intent": "",

                "context": "",

                "tool_result": "",

                "sources": []
            }

            result = st.session_state.graph.invoke(
                state
            )

            answer = result["messages"][-1].content

            sources = result.get(
                "sources",
                []
            )

            st.markdown(answer)

            if sources:

                st.markdown(
                    "**Sources:**"
                )

                for source in sources:

                    st.markdown(
                        f"- `{source}`"
                    )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )