import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast


class ContextAwareChatbot:
    """
    By using a model and tokenizer of the specified types, build a chatbot class by keeping
    recent chat history containing the bot responses and user inputs.

    """
    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast,
                 device: str = None, context_window: int = 5) -> None:
        """

        Args:
            model (LlamaForCausalLM): local model initialized through the transformers library
            tokenizer (LlamaTokenizerFast): local tokenizer initialized through the transformers library
            device (str): either 'cuda' or 'cpu'; if not specified, uses 'cuda' if available
            context_window (str): number of past prompts to consider; default is 5
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.context_window = context_window
        self.history = []
        # Set the model device; input device will be cast for every input in the response method
        self.model.to(self.device)

    def update_history(self, user_input: str = None, bot_response: str = None) -> None:
        """
        Keep track of the recent history in the conversation of the bot with the user.
        Whenever the context window gets larger than twice the window specified, the window is updated; twice is
        justified by keeping, at least, n = context_window messages from both the user and the bot.

        Args:
            user_input: input that will be (or was) sent to the bot as a prompt
            bot_response: bots response to the user input

        Returns:
            None
        """
        if bot_response and user_input:
            self.history.append(f"User: {user_input}")
            self.history.append(f"Bot: {bot_response}")
        elif user_input:
            self.history.append(f"User: {user_input}")
        else:
            self.history.append(f'Bot: {bot_response}')

        if len(self.history) > self.context_window * 2:
            self.history = self.history[-self.context_window * 2:]

    def generate_response(self, user_input: str) -> str:
        """
        Get response to the user_input. Updates the history including the user_input to the current prompt and, next,
        including the response.

        Args:
            user_input: user prompt to the bot

        Returns:
            bot_response (str): bot's reply to user_input
        """
        self.update_history(user_input=user_input)
        input_text = "\n".join(self.history)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        bot_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.update_history(bot_response=bot_response)
        return bot_response


class PersonalityChatbot(ContextAwareChatbot):
    """
    Inheriting the ContextAwareChatbot features, implements possible 'personalities' for the bot, taking form as
    predefined messages sent along with the input as prompts.

    """
    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast,
                 device: str = None, context_window: int = 5) -> None:
        super().__init__(model, tokenizer, device, context_window)
        self.personalities = {
            "friendly": "You are a friendly and approachable chatbot.",
            "professional": "You are a professional and formal chatbot.",
            "humorous": "You are a humorous and witty chatbot."
        }
        self.current_personality = "friendly"

    def set_personality(self, personality: str) -> None:
        """
        Define which personality the bot will take. Standard: friendly.

        Args:
            personality: one of 'friendly', 'professional' or 'humorous'

        Returns:
            None
        """
        if personality in self.personalities:
            self.current_personality = personality
        else:
            raise ValueError("Personality not found")

    def generate_response(self, user_input: str) -> str:
        """
        Redefines parent class' generate_response with the same functionalities adding the personality promt at the
        start of every input.

        Args:
            user_input (str): prompt from the user to the bot

        Returns:
            bot_response (str): bot's reply
        """
        self.update_history(user_input=user_input)
        personality_prompt = self.personalities[self.current_personality]
        input_text = personality_prompt + "\n" + "\n".join(self.history)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        bot_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.update_history(bot_response=bot_response)
        return bot_response


class Chatbot(PersonalityChatbot):
    """
    Inherits the PersonalityChatbot class with no further modifications but adding a response_length limiter to the
    output.

    """
    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast,
                 device: str = None, context_window: int = 5) -> None:
        super().__init__(model, tokenizer, device, context_window)
        self.response_length = 100

    def set_response_length(self, length: int) -> None:
        self.response_length = length
