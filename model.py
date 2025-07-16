from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from agents import Runner, Agent
from dotenv import load_dotenv
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    model=model,
    tracing_disabled=True,
)

llm = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    response = await Runner.run(llm, user_input, run_config=config)
    return {"response": response.final_output}
