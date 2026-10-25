import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END

from .config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K
)

from .state import AgentState
from .vector_store import load_vector_store
from .tools import (
    get_order_status,
    check_refund_eligibility
)

from .prompts import prompt


llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY
)


def detect_intent(state: AgentState):

    question = state["messages"][-1].content

    question_lower = question.lower()

    if any(
        word in question_lower
        for word in [
            "order",
            "delivery",
            "track",
            "where is my"
        ]
    ):

        intent = "order"

    elif any(
        word in question_lower
        for word in [
            "refund",
            "return",
            "money back"
        ]
    ):

        intent = "refund"

    else:

        intent = "general"

    return {
        "intent": intent
    }


def retrieve_context(state: AgentState):

    question = state["messages"][-1].content

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    sources = list(
        dict.fromkeys(
            doc.metadata.get(
                "source",
                "Unknown"
            )
            for doc in documents
        )
    )

    return {
        "context": context,
        "sources": sources
    }


def extract_order_id(text):

    match = re.search(
        r"\b\d{4}\b",
        text
    )

    if match:
        return match.group()

    return None


def execute_tool(state: AgentState):

    question = state["messages"][-1].content

    order_id = extract_order_id(question)

    if not order_id:

        return {
            "tool_result":
            "No order ID was provided."
        }

    if state["intent"] == "order":

        result = get_order_status.invoke(
            {
                "order_id": order_id
            }
        )

    else:

        result = check_refund_eligibility.invoke(
            {
                "order_id": order_id
            }
        )

    return {
        "tool_result": result
    }


def generate_answer(state: AgentState):

    question = state["messages"][-1].content

    formatted_prompt = prompt.invoke(
        {
            "context": state.get(
                "context",
                ""
            ),

            "tool_result": state.get(
                "tool_result",
                ""
            ),

            "question": question
        }
    )

    response = llm.invoke(
        formatted_prompt
    )

    return {
        "messages": [
            response
        ]
    }


def route_after_intent(state: AgentState):

    if state["intent"] in [
        "order",
        "refund"
    ]:

        return "tool"

    return "retrieve"


def build_graph():

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "intent",
        detect_intent
    )

    workflow.add_node(
        "retrieve",
        retrieve_context
    )

    workflow.add_node(
        "tool",
        execute_tool
    )

    workflow.add_node(
        "generate",
        generate_answer
    )

    workflow.set_entry_point(
        "intent"
    )

    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "tool": "tool",
            "retrieve": "retrieve"
        }
    )

    workflow.add_edge(
        "tool",
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()