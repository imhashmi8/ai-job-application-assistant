import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from scraper import scrape_job_posting
import tiktoken

# --- Load environment variables --- #
load_dotenv(override=True)


# --- Model switcher: change to "ollama" to use local model --- #
MODE = "openai" # or "ollama"

if MODE == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = "gpt-4o-mini"
else:
    client = OpenAI(base_url=os.getenv("OLLAMA_BASE_URL", api_key="ollama"))
    MODEL = "llama3.2"

# --- Token counting utility --- #
def count_tokens(text, model=MODEL):
    """Count tokens in a text string for the specified model."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# --- Core function to stream response from the model --- #
def stream_response(messages):
    """"Stream the response from the model."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
        stream=True
    )
    result = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            result += delta
    print()  # for newline after streaming
    return result