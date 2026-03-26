import os

from typing import Annotated, TypedDict, Union
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv

# 1. Define the State (what the agent remembers during the loop)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]

# 2. Define the Nodes (the "brains" of the operation)
def call_model(state: AgentState):
    load_dotenv()
    print(os.getenv('OPENAI_API_KEY')[:5])  # Debug: Check if the key is loaded
    model = ChatOpenAI(model="gpt-4o", streaming=True)
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 3. Define the Graph
workflow = StateGraph(AgentState)

# Add the execution node
workflow.add_node("agent", call_model)

# Set the entry point
workflow.set_entry_point("agent")

# For a simple RAG, we end here; for an Agent, you'd add tool nodes
workflow.add_edge("agent", END)

# Compile the graph
app_agent = workflow.compile()
