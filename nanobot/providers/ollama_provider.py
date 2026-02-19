"""Ollama provider implementation using native ollama library."""

from typing import Any

import ollama

from loguru import logger
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OllamaProvider(LLMProvider):
    """
    Native Ollama provider.
    
    Bypasses LiteLLM to use the official `ollama` Python library, which offers
    better support for local model features and is easier to debug for local setups.
    """
    
    def __init__(
        self, 
        api_base: str | None = None,
        default_model: str = "llama3",
        options: dict[str, Any] | None = None,
    ):
        super().__init__(api_base=api_base)
        self.default_model = default_model
        self.options = options or {}
        self.client = ollama.AsyncClient(host=api_base)
        
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via Ollama native client.
        """
        model = model or self.default_model
        
        # Merge runtime options with config options
        # Note: ollama.chat accepts 'options' dict for parameters like num_ctx, temperature, etc.
        # We start with config defaults
        run_options = self.options.copy()
        
        # Override with explicit call parameters if they differ from defaults
        # standard LLMProvider params map to Ollama options:
        # temperature -> temperature
        # max_tokens -> num_predict
        run_options["temperature"] = temperature
        if max_tokens > 0:
            run_options["num_predict"] = max_tokens

        # Transform messages if needed (Ollama expects role/content, images in specific format)
        # We assume standard OpenAI-like format for now which Ollama mostly accepts
        
        # Transform messages: Ollama expects tool_calls arguments as dict, not JSON string
        ollama_messages = []
        for msg in messages:
            # Create a copy to avoid modifying the original list in place
            new_msg = msg.copy()
            
            if "tool_calls" in new_msg and new_msg["tool_calls"]:
                new_tool_calls = []
                for tc in new_msg["tool_calls"]:
                    if isinstance(tc, dict):
                        new_tc = tc.copy()
                        if "function" in new_tc:
                            func = new_tc["function"]
                            if isinstance(func, dict) and "arguments" in func:
                                args = func["arguments"]
                                if isinstance(args, str):
                                    import json
                                    try:
                                        new_tc["function"] = func.copy()
                                        new_tc["function"]["arguments"] = json.loads(args)
                                    except json.JSONDecodeError:
                                        # If parsing fails, leave as is (though it might fail later)
                                        pass
                        new_tool_calls.append(new_tc)
                    else:
                        new_tool_calls.append(tc)
                new_msg["tool_calls"] = new_tool_calls
            
            if new_msg.get("role") == "tool":
                new_msg.pop("name", None)

            ollama_messages.append(new_msg)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": run_options,
            "stream": False,  # We don't support streaming in basic LLMProvider yet
        }
        
        if tools:
            kwargs["tools"] = tools

        try:
            # Native chat call
            response = await self.client.chat(**kwargs)
            
            return self._parse_response(response)
            
        except Exception as e:
            import json
            # Log the full payload for debugging
            logger.error(f"Ollama payload error: {str(e)}")
            try:
                # Redact potential sensitive info if needed, but for now dump as is for debugging
                logger.debug(f"Ollama payload: {json.dumps(kwargs, default=str)}")
            except:
                pass
                
            return LLMResponse(
                content=f"Error calling Ollama: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse Ollama response into standard LLMResponse."""
        message = response.get("message", {})
        content = message.get("content")
        
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                function = tc.get("function", {})
                tool_calls.append(ToolCallRequest(
                    id=f"call_{len(tool_calls)}", # Ollama doesn't always provide IDs
                    name=function.get("name", ""),
                    arguments=function.get("arguments", {}),
                ))
        
        # cleanup content if it's None but tool calls exist (Ollama might return None content)
        if content is None and tool_calls:
            content = ""
            
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.get("done_reason", "stop"),
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            },
        )

    def get_default_model(self) -> str:
        return self.default_model
