from fastapi import APIRouter, HTTPException, Depends
from genai_chatbot.schemas.preferences import Preferences
from genai_chatbot.models.chatbot import Chatbot
from genai_chatbot.dependencies import get_model

router = APIRouter()


@router.post("/")
async def set_preferences(prefs: Preferences, chatbot: Chatbot = Depends(get_model)):
    try:
        chatbot.set_personality(prefs.personality)
        chatbot.set_response_length(prefs.response_length)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "Preferences updated"}
