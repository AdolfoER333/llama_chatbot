from fastapi import FastAPI
from genai_chatbot.routes import generate, preferences


app = FastAPI()
app.include_router(generate.router, prefix="/generate")
app.include_router(preferences.router, prefix="/set_preferences")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
