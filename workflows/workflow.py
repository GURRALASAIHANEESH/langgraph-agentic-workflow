from langgraph.graph import StateGraph, END

from typing import Annotated, List, TypedDict

from agents.plan_agent import plan_agent
from agents.tool_agent import tool_agent
from agents.refine_agent import refine_tasks
from agents.reflect_agent import reflect_on_results

class GraphState(TypedDict):
    input: str
    results: list[str]
    subtasks: list[str]
    retry_count: int
    done: bool

def add_plan(state: GraphState) -> GraphState:
    subtasks = plan_agent(state["input"])
    return {**state, "subtasks": subtasks}

def run_tool(state: GraphState) -> GraphState:
    task = state["subtasks"][0]
    result = tool_agent(task)
    return {
        **state,
        "results": state["results"] + [result],
        "subtasks": state["subtasks"][1:]
    }

def should_continue(state: GraphState) -> str:
    return "end" if state.get("done") or not state["subtasks"] else "continue"

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("Plan", add_plan)
    builder.add_node("Refine", refine_tasks)
    builder.add_node("Execute", run_tool)
    builder.add_node("Reflect", reflect_on_results)

    builder.set_entry_point("Plan")
    builder.add_edge("Plan", "Refine")
    builder.add_edge("Refine", "Execute")
    builder.add_conditional_edges("Execute", should_continue, {"continue": "Reflect", "end": END})
    builder.add_conditional_edges("Reflect", should_continue, {"continue": "Refine", "end": END})

    return builder.compile()
