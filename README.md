# AI Chatbot with LangChain, Qwen2.5-0.5B, and Gradio

This project is an open-source AI chatbot built using [LangChain](https://github.com/langchain-ai/langchain), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), and [Gradio](https://gradio.app/). It supports conversational memory and a customizable prompt.

## Features

- Uses HuggingFace Transformers for model inference
- Supports conversational memory with LangChain
- Interactive web UI via Gradio
- Easily switch models (Qwen, Mistral, etc.)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AIChatBotLangChainLLM.git
   cd AIChatBotLangChainLLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download your desired model from HuggingFace (e.g. Qwen2.5-0.5B) and set the `model_name` path in `AIChatBot.py`.**

4. **(Optional) Set your HuggingFace token:**
   ```python
   os.environ["HUGGING_FACE_HUB_TOKEN"] = "your_token_here"
   ```

## Usage

Run the chatbot locally:
```bash
python AIChatBot.py
```
Access the Gradio UI at [http://localhost:7860](http://localhost:7860).

## Customization

- Change the prompt in `AIChatBot.py` to fit your use case.
- Adjust model parameters for speed or quality.

## License

This project is open-source under the MIT License.
