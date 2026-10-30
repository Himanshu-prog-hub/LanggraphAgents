from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import HumanMessage,BaseMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages #reducer function to add multiple msgs like how we appended 
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""

    return a+b

@tool
def subtract(a: int, b: int):
    """This is an subtraction function that subtracts 2 numbers a-b if a is larger than b and b-a if b is larger that a"""
    return a-b

tools = [add,subtract]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = "You are my AI Assistant, please anywer my query to the best of your ability")
    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

def should_continue(state : AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools",tool_node)
graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END,
    },
)

graph.add_edge("tools","our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","Add 34 + 21 and then whatever you get as resule subtract that with 10")]}

print_stream(app.stream(inputs,stream_mode="values"))