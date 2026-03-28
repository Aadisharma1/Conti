from __future__ import annotations

from typing import Any


def messages_to_text(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # gpt2 and other old models dont have chat templates
    parts: list[str] = []
    for m in messages:
        role, content = m.get("role", ""), m.get("content", "")
        if role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "".join(parts).strip()


def build_supervised_example(tokenizer, user: str, assistant: str) -> str:
    msgs: list[dict[str, str]] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return messages_to_text(tokenizer, msgs, add_generation_prompt=False)


def user_prompt_only(tokenizer, user: str) -> str:
    msgs: list[dict[str, str]] = [{"role": "user", "content": user}]
    return messages_to_text(tokenizer, msgs, add_generation_prompt=True)


def replay_item_to_text(tokenizer, item: dict[str, Any]) -> str:
    msgs = item.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("replay item needs messages list")
    return messages_to_text(tokenizer, msgs, add_generation_prompt=False)
