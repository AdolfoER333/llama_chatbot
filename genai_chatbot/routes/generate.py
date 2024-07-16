from fastapi import APIRouter, Depends
from genai_chatbot.schemas.user_input import UserInput
from genai_chatbot.models.chatbot import Chatbot
from genai_chatbot.dependencies import get_model

router = APIRouter()


@router.post("/")
async def generate(input_text: UserInput, chatbot: Chatbot = Depends(get_model)):
    response = chatbot.generate_response(input_text.prompt)
    return {"response": response}
