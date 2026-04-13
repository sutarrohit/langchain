import json

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_stock_price, get_weather

load_dotenv()


# Main agent loop
def execute_agent_loop(
    query: str, llm_with_tools: Runnable, tools: list[BaseTool], max_iterations: int = 5
):
    """Execute llm and run tools"""

    iteration = 0
    messages = [HumanMessage(content=query)]

    while iteration < max_iterations:

        iteration += 1
        print(f"🔄 Iteration {iteration} \n")
        print("-" * 70)

        response = llm_with_tools.invoke(messages)
        messages.append(response)
        print("all message =======", messages)

        if not response.tool_calls:
            print(f"\nFinal Answer ====> \n {response.content}")
            break

        for tool_call in response.tool_calls:
            print(f"\n🔧 Tool Call: {tool_call['name']}")
            print(f"   Arguments: {json.dumps(tool_call['args'], indent=2)}")

            tool_fun = next(t for t in tools if t.name == tool_call["name"])
            tool_result = tool_fun.invoke(tool_call["args"])
            print(f"   Result: {json.dumps(tool_result, indent=2)}")

            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result), tool_call_id=tool_call["id"]
                )
            )

    if iteration >= max_iterations:
        print(f"\n⚠️ Max iterations ({max_iterations}) reached")

    print(f"{'='*70}\n")
    return messages


def main():

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    tools = [get_weather, get_stock_price]
    llm_with_tools = llm.bind_tools(tools)

    print("=" * 70)
    query = (
        "What's the weather like in San Francisco, CA? and check the AAPL stock price"
    )

    print(f"\nQuery: {query}")
    execute_agent_loop(query=query, llm_with_tools=llm_with_tools, tools=tools)


if __name__ == "__main__":
    main()
