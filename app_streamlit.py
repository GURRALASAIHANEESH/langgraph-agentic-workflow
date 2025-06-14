import streamlit as st
from workflows.workflow import build_graph

st.set_page_config(page_title="Agentic LangGraph", layout="centered")
st.title("🧠 LangGraph Agentic Workflow")
st.markdown("Enter a task and see how the agent plans, refines, solves, and reflects on it.")

graph = build_graph()
user_input = st.text_area("Enter your query:", height=100)

if st.button("Run Agent Workflow"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke({
                    "input": user_input,
                    "results": [],
                    "subtasks": [],
                    "retry_count": 0,
                    "done": False
                })
                st.subheader("✅ Final Output:")
                for step in result["results"]:
                    st.markdown(f"- {step}")
            except Exception as e:
                st.error(f"🔥 Workflow failed: {e}")

