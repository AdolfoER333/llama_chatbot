import pytest
from genai_chatbot.models.chatbot import Chatbot
from genai_chatbot.utils.load_model import load_model_and_tokenizer


# Tests for the chatbot class only, regarding the personality changes, response length setting and output generation.

@pytest.fixture
def chatbot():
    model, tokenizer = load_model_and_tokenizer()
    chatbot = Chatbot(model, tokenizer)
    return chatbot


def test_generate_response(chatbot):
    user_input = "Hello, how are you?"
    response = chatbot.generate_response(user_input)
    assert isinstance(response, str)


def test_set_personality(chatbot):
    chatbot.set_personality("humorous")
    assert chatbot.current_personality == "humorous"

    with pytest.raises(ValueError):
        chatbot.set_personality("unknown_personality")


def test_set_response_length(chatbot):
    chatbot.set_response_length(150)
    assert chatbot.response_length == 150
