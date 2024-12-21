from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import json

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to format data into SSE format
def format_sse(event: str, data: str) -> str:
    # Replace newlines in the data with '__n__'
    formatted_data = data.replace("\n", "__n__")
    # Format the SSE message
    res = f"event: {event}\ndata: {formatted_data}\n\n"
    return res


async def mock_stream(prompt: str):
    words = prompt.split()  # Split the prompt into words
    for word in words:
        time.sleep(0.5)  # Simulate delay
        yield word  # Mocking each streamed chunk

# Function to stream the SSE response
async def sse_generator():
    prompt = """
    I'm sorry for the confusion, but the information you provided does not specifically mention tax brackets.
    However, I can provide you with a general overview of tax brackets in Nepal:

    - Income up to a certain threshold is taxed at a lower rate
    - Income above that threshold is taxed at a higher rate
    - The tax rates and brackets may vary depending on the individual's income level

    For more detailed information on specific tax brackets in Nepal,
    I recommend consulting the Income Tax Act and relevant tax regulations.
    """
    source = "{}" 

    time.sleep(1)

    # Send the start event
    yield format_sse("start", json.dumps({"start": True}))

    # Stream data chunks
    async for chunk in mock_stream(prompt):
        yield format_sse("data", chunk)

    # Send the end event
    yield format_sse("end", json.dumps({"end": True, "source": source}))

@app.post("/ask")
async def sse_endpoint():
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
