from fastapi.testclient import TestClient
from genai_chatbot.app import app

client = TestClient(app)


def test_generate_response():
    """
    Asserts that a response was given (status code 200); the text 'response' is written in the output; and asserts
    whether the value of the output has the string type.

    Returns:
        None
    """
    response = client.post(
        "/generate/",
        json={"prompt": "Hello, how are you?"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert isinstance(response.json()["response"], str)
