# LangGraph Agentic Workflow App

This project implements an **agentic workflow pipeline** using **LangGraph**, designed to split a user query into subtasks, refine them, solve them using tools, and reflect on the results. It includes:

* Task Planning Agent
* Refinement Agent
* Tool Execution Agent
* Reflection Agent

All components are integrated into a **LangGraph-based pipeline** and deployed via a **Streamlit interface**.

---

## 📦 Features

* 🧠 Task decomposition using LLMs (PlanAgent)
* 🔁 Dynamic task refinement and retry logic (RefineAgent)
* ⚙️ Tool execution using Calculator and Tavily Search (ToolAgent)
* 🪞 Feedback loop to evaluate task completeness (ReflectAgent)
* 📊 Streamlit app for interactive usage
* 🧩 Modular code structure with API key protection using `.env`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/langgraph-agentic-workflow.git
cd langgraph-agentic-workflow
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file with the following:

```env
OPENAI_API_KEY=your_openrouter_or_openai_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
TAVILY_API_KEY=your_tavily_key
```

> 💡 Use `.env.example` as a reference.

### 5. Run the App Locally

```bash
streamlit run app_streamlit.py
```

---

## 🧪 Sample Query

Try typing:

```text
how to create a resume
```

You’ll see:

* Subtasks planned
* Refined iteratively
* Executed with tool support
* Final reflection to validate completeness

---

## 🧠 Project Structure

```
langgraph-agentic-workflow/
├── agents/
│   ├── plan_agent.py
│   ├── refine_agent.py
│   ├── reflect_agent.py
│   └── tool_agent.py
├── workflows/
│   └── workflow.py
├── app_streamlit.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧰 Tools Used

* **LangGraph**: For defining the pipeline
* **LangChain**: LLM and tool integration
* **OpenRouter/OpenAI**: LLM access
* **Tavily**: Web search tool
* **Streamlit**: Web interface

---

## ✅ Evaluation Goals Met

| Goal                        | Status ✅ |
| --------------------------- | -------- |
| LangGraph Workflow          | ✅ Yes    |
| Reliable Subtask Refinement | ✅ Yes    |
| Feedback Loop with Retry    | ✅ Yes    |
| Modular, Readable Code      | ✅ Yes    |
| Streamlit Deployment        | ✅ Yes    |

---

## 📩 Contact / Contribution

For any issues or suggestions, open an issue or reach out via GitHub.

---

> © 2025 – Built for AI Intern Project Evaluation
