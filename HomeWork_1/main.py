"""Ask an OpenRouter model a question; let it call local tools until it can answer."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS, run_tool

DEFAULT_MODEL = "google/gemma-4-31b-it:free"
MAX_TOOL_ROUNDS = 5
DEFAULT_QUESTION = (
    "How many times does the letter 'r' appear in 'strawberry raspberry'? "
    "Multiply that count by 137 and add 2.5."
)


def create_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set (add it to .env).")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def ask(client: OpenAI, model: str, question: str) -> str:
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Use tools for counting and arithmetic."},
        {"role": "user", "content": question},
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto"
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or ""

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            print(f"[tool] {call.function.name}({call.function.arguments}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    raise RuntimeError(f"No final answer after {MAX_TOOL_ROUNDS} tool rounds.")


def main() -> None:
    load_dotenv()
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"[model] {model}\n[question] {question}")
    print(f"[answer] {ask(create_client(), model, question)}")


if __name__ == "__main__":
    main()
