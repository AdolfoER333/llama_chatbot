from pydantic import BaseModel


class Preferences(BaseModel):
    personality: str = "friendly"
    response_length: int = 100
