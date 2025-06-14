import os
os.environ["LANGGRAPH_RECURSION_LIMIT"] = "50"
import streamlit as st
from workflows.workflow import build_graph

# Build LangGraph once
graph = build_graph()

st.set_page_config(page_title="LangGraph Agentic Workflow", layout="centered")

st.title(" Agentic Task Solver")
st.markdown("Built with LangGraph, OpenRouter, and real tools.")

# Text input from user
user_input = st.text_area("Enter your query here:", height=100)

if st.button("Run Workflow"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            result = graph.invoke({"input": user_input})

        st.subheader("Final Results:")
        for step in result["results"]:
            st.markdown(f"- {step}")
