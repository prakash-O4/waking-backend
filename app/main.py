from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.helpers import SupabaseHelper

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Wakil-G"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Pinecone initialization function with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def initialize_pinecone():
    logger.info("Initializing Pinecone connection")
    index_name = "wakil-g"
    try:
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        logger.info("Pinecone connection successful")
        return retriever
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

# QA prompt
qa_system_prompt = """You are an AI assistant name of Wakil-G specialized in answering questions about Nepal's laws and constitution. Use the following examples to guide your responses, then answer the user's question based on the provided context.

Example 1:
Human: What is the official language of Nepal according to the constitution?
Wakil-G: According to Article 7 of the Constitution of Nepal, the official language of Nepal is Nepali in the Devanagari script. The constitution also recognizes all languages spoken as mother tongues in Nepal as national languages.

Example 2:
Human: Can you explain the concept of federalism in Nepal?
Wakil-G: Federalism in Nepal, as established by the 2015 Constitution, divides the country into seven provinces. This system distributes power between the central government and provincial governments. Key points include:

1. Three tiers of government: federal, provincial, and local
2. Each level has specific powers and responsibilities
3. Aim: to promote local autonomy and decentralize governance

For a full understanding of Nepal's federal structure, refer to Part 5 of the Constitution of Nepal.

Example 3:
Human: What are the fundamental rights guaranteed by the Nepal Constitution?
Wakil-G: The Constitution of Nepal guarantees several fundamental rights in Part 3, Articles 16-48. Some key rights include:

1. Right to freedom (Article 17)
2. Right to equality (Article 18)
3. Right to communication (Article 19)
4. Right relating to justice (Article 20)
5. Right of victim of crime (Article 21)
6. Right against torture (Article 22)
7. Right to privacy (Article 28)

Each right has specific provisions and limitations. For a complete list and details, refer to Part 3 of the Constitution of Nepal.

Now, please answer the following question based on the provided context:

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Contextualized question function
def contextualized_question(question: str, chat_history: List[dict]) -> str:
    if not chat_history:
        return question
    
    # history_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    history_prompt = "\n".join([f"{msg}" for msg in chat_history])
    context_prompt = f"""Given the following chat history and the latest user question, 
    formulate a standalone question that can be understood without the chat history:

    Chat History:
    {history_prompt}

    Latest User Question: {question}
 
    Standalone Question:"""
    
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    response = llm.invoke(context_prompt)
    return response.content

# FastAPI models
class QuestionInput(BaseModel):
    question: str

def format_sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

@app.post("/ask")
async def ask_question(input: QuestionInput,authorization: str = Header(None)):
    try:
        supabase_helper = SupabaseHelper()
         # Verify token and get user ID
        user_id =  supabase_helper.get_user_id(authorization)
        
        # Get user's chat history
        chat_histories =  supabase_helper.get_user_chat_history(user_id)

        # Initialize Pinecone for this request
        retriever = initialize_pinecone()
        
        # Get contextualized question
        context_question = contextualized_question(input.question, chat_histories)
        
        # Retrieve relevant document
        retrieved_docs = retriever.invoke(context_question)
        
        if retrieved_docs:
            context = retrieved_docs[0].page_content
            source = retrieved_docs[0].metadata
        else:
            context = "No relevant information found"
            source = "Unknown"

        # Prepare chat history
        chat_history = [
            # HumanMessage(content=msg) if msg["role"] == "human" else 
            AIMessage(content=msg)
            for msg in chat_histories
        ]

        # Prepare the prompt
        prompt = qa_prompt.format(context=context, question=input.question, chat_history=chat_history)

        # Initialize ChatOpenAI with streaming
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True)

        async def generate_sse():
            yield format_sse("start", json.dumps({"start": True}))

            async for chunk in llm.astream(prompt):
                if chunk.content:
                    yield format_sse("data", chunk.content)
            
            yield format_sse("end", json.dumps({"end": True, "source": source}))

        return StreamingResponse(generate_sse(), media_type="text/event-stream")
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        else:
            logger.error(f"Error in /ask endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)