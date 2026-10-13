"""Serve a shared RAG pipeline through FastAPI."""
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from .pipeline import build_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_chain = build_pipeline()
    app.state.query_lock = Lock()
    try:
        yield
    finally:
        app.state.rag_chain = None


app = FastAPI(title="PDF RAG API", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def reject_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class Answer(BaseModel):
    answer: str


@app.post("/ask", response_model=Answer)
def ask(payload: Question, request: Request):
    with request.app.state.query_lock:
        answer = request.app.state.rag_chain.invoke(payload.question)
    return Answer(answer=answer)
