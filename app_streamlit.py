import streamlit as st
import os
from workflows.workflow import build_graph

# ✅ Set page config FIRST
st.set_page_config(page_title="Agentic LangGraph", layout="centered")

# ✅ Now it's safe to use other Streamlit commands
st.title("🧠 LangGraph Agentic Workflow")

# Debug: Show if your API key is loaded (remove before submission)
st.write("🔑 OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))
st.write("🌍 OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE"))

# Your app logic continues here...
graph = build_graph() # Assuming build_graph() is defined elsewhere and returns your LangGraph
user_input = st.text_input("Enter a task and see how the agent plans, refines, solves, and reflects on it.")

if st.button("Run Agent Workflow"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = graph.invoke(
                {
                    "input": user_input,
                    "results": [],
                    "subtasks": [],
                    "retry_count": 0,
                    "done": False
                },
                config={"recursion_limit": 20}
            )

        st.subheader("✅ Final Output:")
        # Check if "results" key exists before iterating, to prevent errors
        if "results" in result:
            for i, step in enumerate(result["results"], 1): # Added 'i' for step numbering
                st.markdown(f"**Step {i}:** {step}") # Used numbering for better clarity
        else:
            st.warning("No results were returned by the agent.")
    else:
        st.warning("Please enter a task before running the workflow.")
