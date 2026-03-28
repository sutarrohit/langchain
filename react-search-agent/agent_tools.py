from langchain.tools import tool
from tavily import TavilyClient

from dotenv import load_dotenv

load_dotenv()


tavily = TavilyClient()

@tool()
def search_tool(query: str) -> str:
    """
    Tool that search over internet

    Args:
      query: the query to search for

    Result :
       The Search Result
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)
