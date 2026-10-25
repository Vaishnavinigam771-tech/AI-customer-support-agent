from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    messages: Annotated[list, add_messages]

    intent: str

    context: str

    tool_result: str

    sources: list