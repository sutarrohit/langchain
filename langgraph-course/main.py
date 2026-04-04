from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)  # sets START -> agent_reason

flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")


def main():
    print("Hello from langgraph!")

    res = app.invoke(
        {
            "messages": [
                HumanMessage(content="What is weather in London? List it and triple it.")
            ]
        }
    )
    
    
    print("=" * 70)
    last = res["messages"][-1].content

    if isinstance(last, list):
        for block in last:
            if block.get("type") == "text":
                print(block["text"])
    else:
        print(last)


if __name__ == "__main__":
    main()
