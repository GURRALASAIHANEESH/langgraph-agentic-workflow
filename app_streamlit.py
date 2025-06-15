import streamlit as st
from workflows.workflow import build_graph

graph = build_graph()

st.set_page_config(page_title="Agentic LangGraph", layout="centered")
st.title("🧠 LangGraph Agentic Workflow")

user_input = st.text_input("Enter a task and see how the agent plans, refines, solves, and reflects on it.")

if st.button("Run Agent Workflow"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = graph.invoke({"input": user_input, "results": [], "subtasks": [], "retry_count": 0, "done": False},config={"recursion_limit": 10})
            st.subheader("✅ Final Output:")
            if "results" in result:
                for i, step in enumerate(result["results"], 1):
                    st.markdown(f"**Step {i}:** {step}") # Corrected indentation here
            else:
                st.warning("No results were returned.")
