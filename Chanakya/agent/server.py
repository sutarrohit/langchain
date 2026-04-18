import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import run_agent_with_event_streaming

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryInput(BaseModel):
    query: str


async def event_generator(user_input: str):
    inputs = {"messages": [{"role": "user", "content": user_input}]}

    async for event in run_agent_with_event_streaming(user_input, stream=True):
        yield event

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/query")
async def process(input: QueryInput):
    return StreamingResponse(event_generator(input.query), media_type="text/event-stream")