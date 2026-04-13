from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from tools import get_stock_price, get_weather

load_dotenv()


def main():

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    tools = [get_weather, get_stock_price]

    agent = create_agent(model=model, tools=tools)

    query = "What's the weather in San Francisco, CA and AAPL stock price?"
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    print(response)
    print("=" * 70)
    print("Final Answer ====>",response["messages"][-1].content)


if __name__ == "__main__":
    main()
