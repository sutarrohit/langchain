import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AgentResponse

from agent_tools import search_tool

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
tools = [search_tool]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(
                "Find 2 lob listing for full stack developer on linked"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
