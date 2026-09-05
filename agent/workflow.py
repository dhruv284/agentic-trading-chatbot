from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    RemoveMessage
)

from typing_extensions import Annotated, TypedDict

from utils.model_loaders import ModelLoader
from toolkit.tools import (
    retriever_tool,
    financials_tool,
    tavilytool
)


SUMMARIZE_AFTER = 6


class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


def content_to_text(content):
    """
    Convert Gemini's structured response content into plain text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content)


class GraphBuilder:

    def __init__(self):

        self.model_loader = ModelLoader()

        self.llm = self.model_loader.load_llm()

        # Use lightweight wrapper tools
        self.tools = [
            retriever_tool,
            financials_tool,
            tavilytool
        ]

        self.llm_with_tools = self.llm.bind_tools(
            tools=self.tools
        )

        self.graph = None


    def _summarize_node(self, state: BotState):

        messages = state["messages"]
        summary = state.get("summary", "")

        # Don't summarize small conversations
        if len(messages) <= SUMMARIZE_AFTER:
            return {}

        messages_to_summarize = messages[:-2]

        if summary:

            summary_prompt = (
                "Update the following conversation summary using the "
                "new messages. Keep it extremely concise. Preserve only "
                "important stock names, financial numbers, user preferences "
                "and decisions.\n\n"
                f"Existing summary:\n{summary}\n\n"
                "New messages:"
            )

        else:

            summary_prompt = (
                "Summarize the following conversation in a very concise "
                "paragraph. Keep only important stock names, financial "
                "numbers, decisions and context."
            )

        summarization_input = (
            messages_to_summarize
            + [HumanMessage(content=summary_prompt)]
        )

        response = self.llm.invoke(
            summarization_input
        )

        # Gemini can return content as a list
        new_summary = content_to_text(
            response.content
        )

        messages_to_delete = [
            RemoveMessage(id=m.id)
            for m in messages_to_summarize
        ]

        return {
            "summary": new_summary,
            "messages": messages_to_delete
        }


    def _chatbot_node(self, state: BotState):

        summary = state.get("summary", "")
        messages = state["messages"]

        system_content = """
You are a trading research assistant.

Use tools appropriately:

1. retriever_tool:
Use this for questions about uploaded documents.

2. financials_tool:
Use this for company financial information.
Example tickers: AAPL, TSLA, MSFT.

3. web_search:
Use this for current news, market events and web information.

Give concise answers.
Do not repeat large amounts of raw tool output.
"""

        if summary:

            system_content += (
                "\n\nPrevious conversation summary:\n"
                + content_to_text(summary)
            )

        system = SystemMessage(
            content=system_content
        )

        # Keep conversation small
        recent_messages = messages[-6:]

        response = self.llm_with_tools.invoke(
            [system] + recent_messages
        )

        # Normalize Gemini text responses
        if isinstance(response.content, list):

            response.content = content_to_text(
                response.content
            )

        return {
            "messages": [response]
        }


    def build(self):

        graph_builder = StateGraph(BotState)

        graph_builder.add_node(
            "summarize",
            self._summarize_node
        )

        graph_builder.add_node(
            "chatnode",
            self._chatbot_node
        )

        tool_node = ToolNode(
            tools=self.tools
        )

        graph_builder.add_node(
            "tools",
            tool_node
        )

        graph_builder.add_edge(
            START,
            "summarize"
        )

        graph_builder.add_edge(
            "summarize",
            "chatnode"
        )

        graph_builder.add_conditional_edges(
            "chatnode",
            tools_condition
        )

        graph_builder.add_edge(
            "tools",
            "chatnode"
        )

        conn = sqlite3.connect(
            "memory.db",
            check_same_thread=False
        )

        checkpointer = SqliteSaver(conn)

        self.graph = graph_builder.compile(
            checkpointer=checkpointer
        )


    def get_graph(self):

        if self.graph is None:
            raise ValueError(
                "Graph is not built. Call build() first."
            )

        return self.graph