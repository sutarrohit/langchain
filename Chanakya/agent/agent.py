import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.scraper import scrape_article
from tools.text_to_speech import text_to_speech_tool

import json

load_dotenv()

# Initialize LLM
model = os.environ.get("MODEL")
llm = ChatGoogleGenerativeAI(model=model, temperature=0)

agent_graph = create_agent(
    model=llm,
    tools=[scrape_article, text_to_speech_tool],
    system_prompt="You are a helpful assistant that can scrape web articles and convert text to speech. When the user provides a URL: 1. First, use the scrape_article tool to scrape the article from the URL. 2. Then, use the text_to_speech_tool to convert the scraped text to speech. When the user provides direct text (not a URL): 1. Use the text_to_speech_tool to convert the provided text to speech. Always use the tools in the above sequence.",
)


# Most detailed: Stream events
async def run_agent_with_event_streaming(user_input: str, stream: bool = False):
    """
    Stream individual events (most granular)
    If stream=True, yields SSE data for FastAPI server.
    """
    inputs = {"messages": [{"role": "user", "content": user_input}]}

    print(f"Starting agent...\n")

    current_llm_tokens = []

    async for event in agent_graph.astream_events(inputs, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                if stream:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                else:
                    print(content, end="", flush=True)
                current_llm_tokens.append(content)

        elif kind == "on_tool_start":
            tool_input = event["data"].get("input")
            if stream:
                yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name'], 'input': tool_input})}\n\n"
            else:
                print(f"\n[Tool Start] Executing tool: {event['name']}")
                print(f"   Input: {tool_input}")

        elif kind == "on_tool_end":
            print(f"[Tool End] Tool finished: {event['name']}")
            output = event["data"].get("output").content

            try:
                if event["name"] == "text_to_speech_tool":
                    output_data = json.loads(output)
                    if stream:
                        yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': output_data})}\n\n"
                    else:
                        if "success" in output_data:
                            print(output_data)
                else:
                    if stream:
                        yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': output})}\n\n"
                    else:
                        print(output)
            except:
                if stream:
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': output})}\n\n"
                else:
                    print(output)



# Example usage
if __name__ == "__main__":
    # Option 3: Stream events (most detailed, async)
    print("\n=== STREAMING EVENTS ===")
    import asyncio

    while True:
        try:
            
            query = input("\n\n Enter URL or Article text : ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not query:
            print("(empty input, try again)")
            continue

        if query.lower() in {"exit", "stop", "q", "quit", "bye"}:
            break

        try:
            asyncio.run(
                run_agent_with_event_streaming(query)
            )
        except Exception as error:
            print("Error : ", error)
            break

