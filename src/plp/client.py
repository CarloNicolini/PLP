"""
LLM Client for PLP.

Wraps LangChain ChatOpenAI to provide a simple interface for LLM calls.
Supports both synchronous and asynchronous invocation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class LLMClient:
    """
    LLM client wrapper using LangChain ChatOpenAI.

    Provides a simple interface for text generation that can be used
    as the stochastic kernel in PLP primitives.

    Example:
        >>> from plp import LLMClient
        >>> llm = LLMClient(model="gpt-4.1-mini", temperature=0.7)
        >>> result = llm.invoke("What is 2+2?")
        >>> print(result)
        4
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        base_url: str = None,
        **kwargs,
    ):
        """
        Initialize the LLM client.

        Args:
            model: OpenAI model name (default: gpt-4.1-mini)
            temperature: Sampling temperature (default: 0.7)
            **kwargs: Additional arguments passed to ChatOpenAI
        """
        self.model = model
        self.temperature = temperature
        self._llm = ChatOpenAI(
            model=model, temperature=temperature, base_url=base_url, **kwargs
        )
        self.name = f"{model}@{temperature}"

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Synchronously generate a response from the LLM.

        Args:
            prompt: The prompt string to send to the LLM
            system_prompt: Optional system prompt to prepend

        Returns:
            The LLM's response as a string
        """
        messages = []
        if system_prompt:
            from langchain_core.messages import SystemMessage

            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))
        response = self._llm.invoke(messages)
        return response.content

    async def ainvoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Asynchronously generate a response from the LLM.

        Args:
            prompt: The prompt string to send to the LLM
            system_prompt: Optional system prompt to prepend

        Returns:
            The LLM's response as a string
        """
        messages = []
        if system_prompt:
            from langchain_core.messages import SystemMessage

            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))
        response = await self._llm.ainvoke(messages)
        return response.content

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, temperature={self.temperature})"
