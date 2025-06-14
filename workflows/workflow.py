from langgraph.graph import StateGraph, END
from langgraph.graph.schema import TypedState
from typing import Annotated, List, TypedDict
from agents.plan_agent import plan_agent
from agents.tool_agent import tool_agent
from agents.refine_agent import refine_agent
from agents.reflect_agent import reflect_on_results

class GraphState(TypedDict):
    input: str
    results: List[str]
    subtasks: List[str]
    retry_count: int
    done: bool

state_schema = TypedState(
    input=str,
    results=Annotated[List[str], lambda x: x or []],
    subtasks=Annotated[List[str], lambda x: x or []],
    retry_count=int,
    done=bool
)

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
    workflow = StateGraph(GraphState)
    workflow.add_node("Plan", add_plan)
    workflow.add_node("Refine", refine_agent)
    workflow.add_node("Execute", run_tool)
    workflow.add_node("Reflect", reflect_on_results)

    workflow.set_entry_point("Plan")
    workflow.add_edge("Plan", "Refine")
    workflow.add_edge("Refine", "Execute")
    workflow.add_conditional_edges("Execute", should_continue, {"continue": "Reflect", "end": END})
    workflow.add_conditional_edges("Reflect", should_continue, {"continue": "Refine", "end": END})

    return workflow.compile()
