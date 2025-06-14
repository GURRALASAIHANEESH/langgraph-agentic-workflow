from workflows.workflow import build_graph

if __name__ == "__main__":
    graph = build_graph()

    user_input = input("Enter your query: ")
    final_state = graph.invoke({"input": user_input})
    print("\nResults:")
    for step in final_state["results"]:
        print(step)
