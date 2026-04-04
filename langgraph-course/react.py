from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool()
def triple(num: float):
    """
    param num: A number to triple
    returns: the triple of the input number
    """
    return num * 3  # ✅ fixed


tools = [TavilySearch(max_results=1), triple]  # ✅ fixed
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
