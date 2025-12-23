from typing import TypedDict, List
from langchain_core.messages import HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class AgentState(TypedDict):
    messages: List[HumanMessage]

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter :")
while user_input != "exit":
    agent.invoke({"messages" : [HumanMessage(content=user_input)]})
    user_input = input("Enter :")
