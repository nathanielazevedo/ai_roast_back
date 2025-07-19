from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
import os
import logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from collections import defaultdict
import time

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)

# In-memory rate limit store
# Structure: {ip: [timestamps]}
rate_limit_store = defaultdict(list)

RATE_LIMIT = 1  # max 3 submissions
WINDOW_SECONDS = 3600  # 1 hour

# Helper to check rate limit
def is_rate_limited(ip: str) -> bool:
    now = time.time()
    timestamps = rate_limit_store[ip]

    # Remove old timestamps
    rate_limit_store[ip] = [ts for ts in timestamps if now - ts < WINDOW_SECONDS]

    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        return True

    rate_limit_store[ip].append(now)
    return False

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model
class AnswerRequest(BaseModel):
    question: str
    answer: str
    mode: str  # 'mentor' or 'drill'

# API endpoint
@app.post("/api/grade")
async def grade_answer(request: Request, payload: AnswerRequest):
    logging.info(f"Incoming request: {payload.dict()}")

    client_ip = request.client.host
    if is_rate_limited(client_ip):
        logging.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Sorry, you reached the limit. You can only submit 3 times per hour. OpenAI isn't cheap."}
        )

    prompt = generate_prompt(payload.question, payload.answer, payload.mode)
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload.answer},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            logging.info(f"AI Response: {reply[:100]}...")  # Log first 100 chars

    except httpx.HTTPStatusError as e:
        logging.error(f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail="Error from OpenAI API.")

    except Exception as e:
        logging.exception("Unexpected error occurred.")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return {"response": reply}

# Prompt generator
def generate_prompt(question: str, answer: str, mode: str) -> str:
    if mode == "drill":
        tone = "Respond harshly and sarcastically, but still provide the correct answer."
    else:
        tone = "Respond kindly and encouragingly, offering constructive feedback."

    return (
        f"You are an AI coding instructor.\n\n"
        f"Question: {question}\n"
        f"Student's Answer: {answer}\n\n"
        f"{tone}"
    )
