from typing import Annotated,TypedDict, Sequence
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a : int, b : int):
    """This is a helper function that adds two integers and returns the result"""
    return a + b

tools = [add]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
).bind_tools(tools)

def agent(state : AgentState) -> AgentState:
    system_message = SystemMessage(content="You are a helpfull ai assistant")
    ai_msg = llm.invoke([system_message, *state["messages"]])

    #exactly the same as:
    # extend 
    # msgs = [system_message]
    # msgs.extend(state["messages"])

    return {"messages" : ai_msg}

def should_continue(state : AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "continue"
    return "end"
        

tool_node = ToolNode(tools = tools)

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "end" : END,
        "continue" : "tools"

    }
)
graph.add_edge("tools", "agent")

app = graph.compile()

response = app.invoke({"messages" : [("user","what is the sum of 1 and 2")]})

print(response)





