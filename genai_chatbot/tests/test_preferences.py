from fastapi.testclient import TestClient
from genai_chatbot.app import app

client = TestClient(app)


def test_set_preferences():
    """
    Tests for personality and response length changing as expected and outputting the update message.

    Returns:
        None
    """
    response = client.post(
        "/set_preferences/",
        json={"personality": "humorous", "response_length": 150}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "Preferences updated"


def test_set_invalid_personality():
    """
    Asserts the input of a personality that does not exist returns an error status code and the expected message.

    Returns:
        None
    """
    response = client.post(
        "/set_preferences/",
        json={"personality": "unknown_personality", "response_length": 150}
    )
    assert response.status_code == 400
    assert "detail" in response.json()
    assert response.json()["detail"] == "Personality not found"
