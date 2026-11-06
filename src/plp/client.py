"""LLM clients for PLP: OpenAI Chat Completions and local Hugging Face Transformers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Protocol, TypeVar

import torch
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

_T = TypeVar("_T", bound=BaseModel)


def normalize_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Convert PLP / LangChain-style message objects into OpenAI chat messages.

    Accepts dicts with ``role`` / ``content``, or objects named like
    ``SystemMessage``, ``HumanMessage``, ``AIMessage`` (duck-typed ``content``).
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and "role" in m:
            out.append(
                {
                    "role": str(m["role"]),
                    "content": m.get("content", "") or "",
                }
            )
            continue
        cls = type(m).__name__
        content = getattr(m, "content", str(m))
        if cls == "SystemMessage":
            out.append({"role": "system", "content": content})
        elif cls == "HumanMessage":
            out.append({"role": "user", "content": content})
        elif cls == "AIMessage":
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": "user", "content": content})
    return out


class PLPChatClient:
    """
    Thin wrapper around :class:`openai.OpenAI` and :class:`openai.AsyncOpenAI`.

    Exposes :meth:`invoke` for synchronous PLP primitives and :meth:`ainvoke`
    for async code paths. ``ChatOpenAI`` is kept as an alias for backwards
    compatibility.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.7,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        client_kwargs = dict(kwargs)
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._sync = OpenAI(**client_kwargs)
        self._async = AsyncOpenAI(**client_kwargs)

    @property
    def model_name(self) -> str:
        """Alias for ``model`` (compatible with older LangChain-style access)."""
        return self.model

    def invoke(self, messages: list[Any]) -> Any:
        """Synchronous chat completion; returns an object with ``content``."""
        payload = normalize_chat_messages(messages)
        resp = self._sync.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content or ""
        return SimpleNamespace(content=text)

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Async chat completion; returns an object with ``content``."""
        payload = normalize_chat_messages(messages)
        resp = await self._async.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content or ""
        return SimpleNamespace(content=text)

    def invoke_json(self, messages: list[Any]) -> dict:
        """Synchronous chat completion with ``response_format=json_object``."""
        payload = normalize_chat_messages(messages)
        resp = self._sync.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)

    async def ainvoke_json(self, messages: list[Any]) -> dict:
        """Async chat completion with ``response_format=json_object``."""
        payload = normalize_chat_messages(messages)
        resp = await self._async.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)

    def invoke_parsed(
        self, messages: list[Any], response_format: type[_T]
    ) -> _T:
        """Structured-output chat completion returning a parsed Pydantic model.

        Thin wrapper around ``chat.completions.parse``. Use when a caller
        needs a typed spec back from the model (e.g. the Circuit Controller's
        ``CircuitSpec``).
        """
        payload = normalize_chat_messages(messages)
        resp = self._sync.chat.completions.parse(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            response_format=response_format,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise ValueError(
                f"Structured output parse returned None for {response_format.__name__}"
            )
        return parsed


class HuggingFaceLocalBackend(Protocol):
    """Loaded causal LM and tokenizer pair (any object with ``tokenizer`` and ``model``)."""

    tokenizer: Any
    model: Any


class LocalHuggingFaceClient:
    """
    Local text generation via Hugging Face ``transformers`` (chat template + ``generate``).

    Implements the same ``invoke`` / ``ainvoke`` surface as :class:`PLPChatClient` so it
    works with :func:`plp.primitives.sample` and :func:`plp.primitives.asample`. Pass any
    object exposing ``tokenizer`` and ``model`` (e.g. a small wrapper around
    ``AutoModelForCausalLM`` / ``AutoTokenizer``).
    """

    def __init__(
        self,
        backend: HuggingFaceLocalBackend,
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 150,
    ) -> None:
        self._backend = backend
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = getattr(backend.model, "name_or_path", None) or "hf-local"

    @property
    def model_name(self) -> str:
        return str(self.model)

    def invoke(self, messages: list[Any]) -> Any:
        """Synchronous generation; returns an object with ``content``."""
        chat_messages = normalize_chat_messages(messages)
        prompt = self._backend.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._backend.tokenizer(prompt, return_tensors="pt").to(
            self._backend.model.device
        )

        with torch.no_grad():
            outputs = self._backend.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._backend.tokenizer.eos_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        text = self._backend.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        return SimpleNamespace(content=text)

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Async wrapper: runs :meth:`invoke` in a thread pool."""
        return await asyncio.to_thread(self.invoke, messages)


# Backwards-compatible name for code and docs that imported LangChain's ChatOpenAI.
ChatOpenAI = PLPChatClient
