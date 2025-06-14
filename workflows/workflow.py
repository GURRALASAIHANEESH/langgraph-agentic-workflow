import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END
from agents.plan_agent import plan_agent
from agents.tool_agent import tool_agent
from agents.refine_agent import refine_tasks
from agents.reflect_agent import reflect_on_results



# Define the schema for your state
class AgentState(TypedDict):
    input: str
    subtasks: List[str]
    results: List[str]
    done: bool
    retry_count: int 

# PlanAgent function
def add_plan(state: AgentState) -> AgentState:
    subtasks = plan_agent(state["input"])
    return {"input": state["input"], "subtasks": subtasks, "results": [], "done": False}

# ToolAgent function
def execute_next_task(state: AgentState) -> AgentState:
    if not state["subtasks"]:
        state["done"] = True
        return state
    
    task = state["subtasks"].pop(0)
    result = tool_agent(task)
    state["results"].append(result)
    return state

# Build LangGraph
def build_graph():
    builder = StateGraph(state_schema=AgentState)

    builder.add_node("Plan", add_plan)
    builder.add_node("Refine", refine_tasks)
    builder.add_node("Execute", execute_next_task)
    builder.add_node("Reflect", reflect_on_results)  

    builder.set_entry_point("Plan")
    builder.add_edge("Plan", "Refine")
    builder.add_edge("Refine", "Execute")

    # Loop execution till subtasks are empty
    builder.add_conditional_edges(
        "Execute",
        lambda state: "Reflect" if state.get("done") else "Execute"
    )

    # Final check in Reflect node
    builder.add_conditional_edges(
        "Reflect",
        lambda state: END if state.get("done") else "Plan"
    )

    return builder.compile()
