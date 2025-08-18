import os
import numpy as np  # add this import

# os.environ["HUGGING_FACE_HUB_TOKEN"] = "Paste your HuggingFace Token Here"

import gradio as gr
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate  # new import

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # corrected import

import torch

# Load an open-source LLM (Qwen2.5-0.5B for faster inference)
model_name = r"Model_Name e.g (Qwen/Qwen2.5-0.5B or Path Containing Model Files Downloaded from Huggingface"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,  # reduced for faster inference
    repetition_penalty=1.1,
    model_kwargs={
        "temperature": 0.7,  # faster, more deterministic output
        # "do_thinking": False  # add if supported by Qwen2.5
    }
)

# If still slow, consider using a smaller model (e.g. Qwen/Qwen1-0.5B) or quantized weights.

# Define a minimal system prompt
custom_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are an AI assistant providing helpful and concise responses.
{history}
User: {input}
AI:"""
)

# Wrap in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
memory = ConversationBufferMemory()

# Use ConversationChain as per imports
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt
)


# Define chatbot function for Gradio ChatInterface
def chat_with_ai(message, history):
    # Save user/assistant pairs to memory
    for i in range(0, len(history) - 1, 2):
        user_msg = history[i]
        bot_msg = history[i + 1]
        if user_msg["role"] == "user" and bot_msg["role"] == "assistant":
            memory.save_context(
                {"input": user_msg["content"]},
                {"output": bot_msg["content"]}
            )
    response = conversation.invoke(message)
    if isinstance(response, dict):
        answer = response.get("response") or response.get("answer") or str(response)
    else:
        answer = str(response)
    return {"role": "assistant", "content": answer}

# Create a Gradio interface
chat_interface = gr.ChatInterface(
    fn=chat_with_ai,
    title="🤖 AI Chatbot with Memory (Qwen 2.5 0.5B)",
    description="An open-source LLM chatbot using LangChain, Memory & Gradio",
    theme="default",  # use built-in theme to avoid error
    type="messages"  # explicitly set messages format
)

# Launch the chatbot
chat_interface.launch()