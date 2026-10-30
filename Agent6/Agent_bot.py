from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

userInput = input("Enter: ")
while userInput != "exit": 
    agent.invoke({"messages": [HumanMessage(content=userInput)]})
    userInput = input("Enter: ")
