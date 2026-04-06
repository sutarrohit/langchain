from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()
