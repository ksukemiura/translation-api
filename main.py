from enum import StrEnum

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel


class Language(StrEnum):
    JAPANESE = "Japanese"
    ENGLISH = "English"
    CHINESE = "Chinese"


class Model(StrEnum):
    GPT_6_LUNA = "gpt-6-luna"
    GPT_6_1_SOL = "gpt-6.1-sol"


class TranslationRequest(BaseModel):
    text: str
    target_language: Language
    model: Model


class TranslationResponse(BaseModel):
    translation: str
    target_language: Language
    model: Model


app = FastAPI()
load_dotenv()
client = AsyncOpenAI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/translate")
async def translate(request: TranslationRequest) -> TranslationResponse:
    response = await client.responses.parse(
        model=request.model,
        instructions=f"You are a translator. Translate the given text into {request.target_language}.",
        input=request.text,
        text_format=TranslationResponse,
    )
    translation = response.output_parsed
    if translation is None:
        raise HTTPException(
            status_code=502,
            detail="Translation could not be parsed"
        )
    return TranslationResponse(
        translation=translation.translation,
        target_language=request.target_language,
        model=request.model,
    )
