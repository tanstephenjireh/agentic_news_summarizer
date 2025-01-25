from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import operator

from lggraph.state import run_oracle
from lggraph.state import run_tool
from lggraph.state import router

from agents.web_search import web_search
from agents.article_content import article_content
from agents.final_answer import final_answer
from agents.chitchat_answer import chitchat_answer

from semtic_router.sr import semantic_layer

tools=[
    web_search,
    article_content,
    final_answer,
    chitchat_answer
]

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

def graph(query, chat_history):
    graph = StateGraph(AgentState)

    graph.add_node("oracle", run_oracle)
    graph.add_node("web_search", run_tool)
    graph.add_node("article_content", run_tool)
    graph.add_node("final_answer", run_tool)
    graph.add_node("chitchat_answer", run_tool)

    graph.set_entry_point("oracle")

    graph.add_conditional_edges(
        source="oracle",  # where in graph to start
        path=router,  # function to determine which node is called
    )
    
    # create edges from each tool back to the oracle
    for tool_obj in tools:
        # if tool_obj.name != "final_answer":
        if tool_obj.name != "final_answer" and tool_obj.name != "chitchat_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # if anything goes to final answer, it must then move to END
    graph.add_edge("final_answer", END)
    graph.add_edge("chitchat_answer", END)
    # graph.add_edge("oracle", END)

    runnable = graph.compile()

    return runnable.invoke({
        "input": semantic_layer(query),
        "chat_history": chat_history
    })