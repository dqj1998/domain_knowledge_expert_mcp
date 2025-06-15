import requests
import os

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

def generate_answer_with_context(prompt: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are an expert in the domain."},
        {"role": "user", "content": f"{context}\n\nUser question: {prompt}"}
    ]
    response = requests.post(
        f"{OPENAI_API_BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": OPENAI_MODEL, "messages": messages}
    )
    return response.json()["choices"][0]["message"]["content"]

def should_follow_link(parent_url: str, link: str) -> bool:
    prompt = f"Should I follow this link to better understand the domain?\nParent: {parent_url}\nLink: {link}"
    return "yes" in generate_answer_with_context(prompt, "")
