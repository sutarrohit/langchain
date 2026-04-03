from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool()
def triple(num: float):
    """
    param num : A number to triple
    returns : the triple o the input number

    Returns:
        _type_: _description_
    """
    return float * 3


tools = [TavilySearch(max_result=1), triple]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
