# Text Generation Chatbot

Simple chatbot using a pre-trained local LLM. The model must be stored in the `pretrained_model/` directory, 
along with its related files, such as the `tokenizer.json`, `config.json` and so on. Implemented chatbots were
a (short) context aware bot and a personality changing bot, with its results available through a local FastAPI endpoint
along with an endpoint for preferences, those being the 3 available personalities and the output length.

The model used for testing this repository was Meta's Llama-2 7B Chat from huggingface, which can be found at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf.

Testing with an Nvidia RTX GeForce 3070Ti 8 GB and 32 GB of RAM resulted in roughly 4 minutes/response. That would not
be acceptable for a production app, but served its study project purpose.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the FastAPI app:
    ```bash
   uvicorn genai_chatbot.app:app --reload

4. Access the endpoints:
   - Response generation
   ```bash
   URL: "http://127.0.0.1:8000/generate" 
   Header: "application/json"
   Body: '{"prompt": "Hello, how are you?"}'
   ```
   - Preferences setting
   ```bash
   URL: "http://127.0.0.1:8000/set_preferences"
   Header: "application/json"
   Body: '{"personality": "humorous", "response_length": 150}'
   ```