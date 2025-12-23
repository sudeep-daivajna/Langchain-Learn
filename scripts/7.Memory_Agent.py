from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]

def process(state : AgentState) -> AgentState:
    response = llm.invoke(input=state["messages"])
    state["messages"].append(AIMessage(content = response.content))
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

user_input = input("Enter : ")

conversation_history: List[Union[HumanMessage, AIMessage]] = []

while True:
    user_input = input("Enter : ")
    if user_input.strip().lower() == "exit":
        break

    conversation_history.append(HumanMessage(content=user_input))

    final_state = agent.invoke({"messages": conversation_history})
    conversation_history = final_state["messages"]  # <-- use returned state

    print(conversation_history[-1].content)