from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Model definitions
class ChatRequest(BaseModel):
    question: str
    thread_id: str
    session_id: str = None

class ChatResponse(BaseModel):
    sender: str
    message: str
    type: str

# Dummy temp chat storage
temp_chat_storage = {}

async def process_request(question, thread_id, temp_chat_id=None, doc_context=None, run_id=None, child_id=None):
    """Mock function to simulate different response types"""
    # Create a dummy response based on keywords in the question
    question_lower = question.lower()
    
    if "social" in question_lower or "story" in question_lower:
        return {
            "type": "social_story",
            "title": "Mock Social Story",
            "content": "This is a mock social story content.",
            "user_guidance": "Here's a social story about handling social situations."
        }
    elif "timer" in question_lower:
        return {
            "type": "timer",
            "title": "Mock Timer",
            "duration": 300,
            "user_guidance": "I've set up a timer for you."
        }
    elif "speech" in question_lower or "deck" in question_lower:
        return {
            "type": "speech_deck",
            "title": "Mock Speech Deck",
            "cards": ["Card 1", "Card 2", "Card 3"],
            "user_guidance": "Here's a speech deck to help you practice."
        }
    elif "chat story" in question_lower:
        return {
            "type": "chat_story",
            "title": "Mock Chat Story",
            "content": "This is a mock chat story.",
            "user_guidance": "Here's an interactive chat story for you."
        }
    elif "confirm" in question_lower:
        return {
            "type": "confirmation",
            "user_guidance": "I need to confirm something with you. Is that okay?"
        }
    else:
        return {
            "type": "direct",
            "content": f"This is a mock direct response to your question: '{question}'"
        }

def get_existing_temp_chat_id_text(thread_id, session_id):
    """Mock function to get existing temp chat ID"""
    if thread_id in temp_chat_storage and 'session_id' in temp_chat_storage[thread_id]:
        if temp_chat_storage[thread_id]['session_id'] == session_id:
            return temp_chat_storage[thread_id]['temp_chat_id'], temp_chat_storage[thread_id]['run_id'], session_id
    return None, None, None

def store_temp_chat_text(thread_id, question, run_id, session_id):
    """Mock function to store temp chat"""
    temp_chat_id = str(uuid.uuid4())
    temp_chat_storage[thread_id] = {
        'temp_chat_id': temp_chat_id,
        'run_id': run_id,
        'session_id': session_id,
        'messages': [{"role": "human", "text": question}]
    }
    return temp_chat_id

def fetch_temp_chat_messages_text(thread_id, temp_chat_id):
    """Mock function to fetch temp chat messages"""
    if thread_id in temp_chat_storage:
        return temp_chat_storage[thread_id].get('messages', [])
    return []

def append_message(temp_chat_id, thread_id, message, role="human"):
    """Mock function to append message to temp chat"""
    if thread_id in temp_chat_storage:
        temp_chat_storage[thread_id]['messages'].append({"role": role, "text": message})
        return True
    return False

def get_agents_topic_id(child_id, topic_name, agent_name):
    """Mock function to get topic ID"""
    return str(uuid.uuid4())

def fetch_selected_child(uid):
    """Mock function to fetch selected child"""
    return "mock_child_id"

def fetch_selected_documents(uid):
    """Mock function to fetch selected documents"""
    return "mock_document_context"

async def get_user_uid_from_google(token):
    """Mock function to validate token"""
    # For testing, allow any token
    if token == "invalid_token":
        return None
    return "mock_user_id"

@app.post("/global_chat")
async def global_chat(
    chat_request: ChatRequest,
    authorization: str = Header(None, description="Bearer token for authentication")
):
    async def chat_stream():
        try:
            await asyncio.sleep(1.4)  # Simulate some delay for processing
            # Validate authorization header
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")
            token = authorization.split(" ")[1]
            uid = await get_user_uid_from_google(token)
            if not uid:
                raise HTTPException(status_code=401, detail="Invalid token")

            child_id = fetch_selected_child(uid)
            if not child_id:
                raise HTTPException(status_code=400, detail="No child selected for user")

            # Initialize temp_chat_id and run_id
            temp_chat_id = None
            run_id = str(uuid.uuid4())
            chat_history = None

            # Check for existing temp_chat_id
            logging.info(f"Checking temp_chat: thread_id={chat_request.thread_id}, session_id={chat_request.session_id}")
            temp_chat_id, stored_run_id, stored_session_id = get_existing_temp_chat_id_text(chat_request.thread_id, chat_request.session_id)

            # Reuse temp_chat_id if found and session_id matches
            if temp_chat_id and stored_session_id == chat_request.session_id:
                run_id = stored_run_id
                chat_history = fetch_temp_chat_messages_text(chat_request.thread_id, temp_chat_id)
                if not chat_history:
                    logging.warning(f"Empty chat history: thread_id={chat_request.thread_id}, temp_chat_id={temp_chat_id}")
                    chat_history = [{"role": "human", "text": chat_request.question}]
                else:
                    chat_history.append({"role": "human", "text": chat_request.question})
                if not append_message(temp_chat_id, chat_request.thread_id, chat_request.question, role="human"):
                    raise HTTPException(status_code=500, detail="Failed to append user message")
                logging.info(f"Reusing temp_chat_id={temp_chat_id}, run_id={run_id}, session_id={chat_request.session_id}")
            else:
                logging.info(f"Creating new temp_chat: thread_id={chat_request.thread_id}, session_id={chat_request.session_id}")
                temp_chat_id = store_temp_chat_text(chat_request.thread_id, chat_request.question, run_id, chat_request.session_id)
                if not temp_chat_id:
                    raise HTTPException(status_code=500, detail="Failed to initialize chat session")
                chat_history = [{"role": "human", "text": chat_request.question}]
                logging.info(f"Created temp_chat_id={temp_chat_id}, run_id={run_id}, session_id={chat_request.session_id}")

            # Process the request with chat history
            agentic_response = await process_request(
                "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history]),
                chat_request.thread_id,
                temp_chat_id=temp_chat_id,
                doc_context="mock context",
                run_id=run_id,
                child_id=child_id
            )

            # Yield user's message
            resp = ChatResponse(sender="you", message=chat_request.question, type="stream")
            yield f"event: stream\ndata: {resp.json()}\n\n"
            await asyncio.sleep(0.1)  # Add a small delay to simulate network

            start_resp = ChatResponse(sender="bot", message="", type="start")
            yield f"event: start\ndata: {start_resp.json()}\n\n"
            await asyncio.sleep(0.1)

            # Handle different response types
            response_type = agentic_response.get("type")
            
            if response_type == "confirmation":
                # Store the response in mock storage
                append_message(temp_chat_id, chat_request.thread_id, agentic_response["user_guidance"], "ai")
                
                # Stream the response without label
                words = agentic_response["user_guidance"]
                for char in words:
                    stream_resp = ChatResponse(sender="bot", message=char, type="stream")
                    yield f"event: stream\ndata: {stream_resp.json()}\n\n"
                    await asyncio.sleep(0.01)  # Simulate typing
                yield f"event: end\ndata: {ChatResponse(sender='bot', message='', type='end').json()}\n\n"
                await asyncio.sleep(0.1)
                yield f"event: session\ndata: {json.dumps({'session_id': chat_request.session_id, 'temp_chat_id': temp_chat_id, 'reuse': True})}\n\n"
                await asyncio.sleep(0.1)

            elif response_type == "direct":
                words = agentic_response["content"]
                for char in words:
                    stream_resp = ChatResponse(sender="bot", message=char, type="stream")
                    yield f"event: stream\ndata: {stream_resp.json()}\n\n"
                    await asyncio.sleep(0.01)  # Simulate typing
                yield f"event: end\ndata: {ChatResponse(sender='bot', message='', type='end').json()}\n\n"
                await asyncio.sleep(0.1)
                yield f"event: session\ndata: {json.dumps({'session_id': chat_request.session_id, 'temp_chat_id': temp_chat_id, 'reuse': False})}\n\n"
                await asyncio.sleep(0.1)

            elif response_type in ["social_story", "timer", "speech_deck", "chat_story"]:
                response_message = json.dumps(agentic_response)
                agent_name = agentic_response.get("type")
                topic_name = agentic_response.get("title")
                user_guidance = agentic_response.get("user_guidance")
                
                if not topic_name:
                    error_resp = ChatResponse(sender="bot", message="Invalid response: missing title", type="error")
                    yield f"event: error\ndata: {error_resp.json()}\n\n"
                    await asyncio.sleep(0.1)
                    return
                    
                topic_name = topic_name.capitalize()
                topic_id = get_agents_topic_id(child_id, topic_name, agent_name)

                yield f"event: {agent_name}\ndata: {ChatResponse(sender='bot', message=response_message, type=agent_name).json()}\n\n"
                await asyncio.sleep(0.1)
                
                if user_guidance:
                    words = user_guidance
                    for char in words:
                        stream_resp = ChatResponse(sender="bot", message=char, type="stream")
                        yield f"event: stream\ndata: {stream_resp.json()}\n\n"
                        await asyncio.sleep(0.01)  # Simulate typing
                
                if topic_id:
                    topic_info = ChatResponse(sender='bot', message=json.dumps({"topic_id": topic_id}), type=agent_name)
                    yield f"event: {agent_name}\ndata: {topic_info.json()}\n\n"
                    await asyncio.sleep(0.1)
                    
                yield f"event: end\ndata: {ChatResponse(sender='bot', message='', type='end').json()}\n\n"
                await asyncio.sleep(0.1)
                yield f"event: session\ndata: {json.dumps({'session_id': chat_request.session_id, 'temp_chat_id': temp_chat_id, 'reuse': False})}\n\n"
                await asyncio.sleep(0.1)

            else:
                error_resp = ChatResponse(sender="bot", message="There was an issue, please try again!", type="error")
                yield f"event: error\ndata: {error_resp.json()}\n\n"
                await asyncio.sleep(0.1)

        except HTTPException as he:
            yield f"event: error\ndata: {ChatResponse(sender='bot', message=he.detail, type='error').json()}\n\n"
        except Exception as e:
            logging.error(f"SSE error: thread_id={chat_request.thread_id}, session_id={chat_request.session_id}, error={e}")
            error_resp = ChatResponse(sender="bot", message=f"Error: {str(e)}", type="error")
            yield f"event: error\ndata: {error_resp.json()}\n\n"

    return StreamingResponse(chat_stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"message": "Welcome to the SSE Chat API!"}

# For testing, if running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)