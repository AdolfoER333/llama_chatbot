from genai_chatbot.utils.load_model import load_model_and_tokenizer
from genai_chatbot.models.chatbot import Chatbot

MODEL, TOKENIZER = load_model_and_tokenizer()


def get_model():
    return Chatbot(MODEL, TOKENIZER)
