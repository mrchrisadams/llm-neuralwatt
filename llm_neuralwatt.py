import llm
from llm.default_plugins.openai_models import _Shared
from llm.default_plugins.openai_models import (
    remove_dict_none_values,
    combine_chunks,
)
import openai
from openai._streaming import SSEDecoder, ServerSentEvent
import json


class EnergyCapturingSSEDecoder(SSEDecoder):
    """
    SSE Decoder that captures Neuralwatt energy comments.
    
    Neuralwatt sends energy data as an SSE comment (`: energy {...}`) just
    before the [DONE] marker. The standard SSEDecoder ignores comments per
    the SSE spec, so we override decode() to capture energy data.
    """
    
    def __init__(self):
        super().__init__()
        self.energy_data = None
    
    def decode(self, line: str) -> ServerSentEvent | None:
        # Check for energy comment before the parent ignores it
        if line.startswith(": energy "):
            energy_json = line[9:]  # Skip ": energy "
            try:
                self.energy_data = json.loads(energy_json)
            except json.JSONDecodeError:
                pass
            return None  # Don't emit as an event
        
        # Fall back to standard SSE decoding
        return super().decode(line)


class NeuralWattOpenAI(openai.OpenAI):
    """
    OpenAI client subclass that captures Neuralwatt energy data from streams.
    
    Overrides _make_sse_decoder() to use EnergyCapturingSSEDecoder, which
    captures the energy comment that Neuralwatt sends before [DONE].
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_decoder = None
    
    def _make_sse_decoder(self):
        self._last_decoder = EnergyCapturingSSEDecoder()
        return self._last_decoder
    
    def get_last_energy_data(self):
        """Get energy data captured from the last streaming request."""
        if self._last_decoder:
            return self._last_decoder.energy_data
        return None


class NeuralWattAsyncOpenAI(openai.AsyncOpenAI):
    """
    AsyncOpenAI client subclass that captures Neuralwatt energy data from streams.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_decoder = None
    
    def _make_sse_decoder(self):
        self._last_decoder = EnergyCapturingSSEDecoder()
        return self._last_decoder
    
    def get_last_energy_data(self):
        """Get energy data captured from the last streaming request."""
        if self._last_decoder:
            return self._last_decoder.energy_data
        return None


@llm.hookimpl
def register_models(register):
    # Register NeuralWatt models with full energy tracking
    register(
        NeuralWattChat(
            "neuralwatt/deepseek-coder-33b-instruct",
            model_name="deepseek-ai/deepseek-coder-33b-instruct",
            api_base="https://api.neuralwatt.com/v1",
        ),
        NeuralWattAsyncChat(
            "neuralwatt/deepseek-coder-33b-instruct",
            model_name="deepseek-ai/deepseek-coder-33b-instruct",
            api_base="https://api.neuralwatt.com/v1",
        ),
        aliases=("neuralwatt-deepseek-coder",),
    )
    register(
        NeuralWattChat(
            "neuralwatt/gpt-oss-20b",
            model_name="openai/gpt-oss-20b",
            api_base="https://api.neuralwatt.com/v1",
        ),
        NeuralWattAsyncChat(
            "neuralwatt/gpt-oss-20b",
            model_name="openai/gpt-oss-20b",
            api_base="https://api.neuralwatt.com/v1",
        ),
        aliases=("neuralwatt-gpt-oss",),
    )
    register(
        NeuralWattChat(
            "neuralwatt/Qwen3-Coder-480B-A35B-Instruct",
            model_name="Qwen/Qwen3-Coder-480B-A35B-Instruct",
            api_base="https://api.neuralwatt.com/v1",
        ),
        NeuralWattAsyncChat(
            "neuralwatt/Qwen3-Coder-480B-A35B-Instruct",
            model_name="Qwen/Qwen3-Coder-480B-A35B-Instruct",
            api_base="https://api.neuralwatt.com/v1",
        ),
        aliases=("neuralwatt-qwen3-coder",),
    )


class NeuralWattShared(_Shared):
    def __init__(self, *args, **kwargs):
        # Handle the case where api_key_name might be passed as a kwarg
        api_key_name = kwargs.pop("api_key_name", None)
        super().__init__(*args, **kwargs)
        # Set up NeuralWatt-specific configuration
        if api_key_name:
            self.needs_key = api_key_name
        else:
            self.needs_key = "neuralwatt"
        self.key_env_var = "NEURALWATT_API_KEY"

    def __str__(self):
        return "Neuralwatt: {}".format(self.model_id)
    
    def get_client(self, key, *, async_=False):
        """
        Get an OpenAI client configured for Neuralwatt.
        
        Uses NeuralWattOpenAI/NeuralWattAsyncOpenAI subclasses that capture
        energy data from streaming responses.
        """
        import os
        from llm.utils import logging_client
        
        kwargs = {}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        if self.needs_key:
            kwargs["api_key"] = self.get_key(key)
        else:
            kwargs["api_key"] = "DUMMY_KEY"
        if self.headers:
            kwargs["default_headers"] = self.headers
        if os.environ.get("LLM_OPENAI_SHOW_RESPONSES"):
            kwargs["http_client"] = logging_client()
        
        if async_:
            return NeuralWattAsyncOpenAI(**kwargs)
        else:
            return NeuralWattOpenAI(**kwargs)


class NeuralWattChat(NeuralWattShared, llm.KeyModel):
    default_max_tokens = None

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """
        Execute a chat completion sending a request to the NeuralWatt API,
        preserving returned energy data to store in llm's local logs.db.
        """
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None

        if stream:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            tool_calls = {}

            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta:
                    for tool_call in chunk.choices[0].delta.tool_calls or []:
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ""
                        index = tool_call.index
                        if index not in tool_calls:
                            tool_calls[index] = tool_call
                        else:
                            tool_calls[
                                index
                            ].function.arguments += tool_call.function.arguments
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            
            # Combine chunks and add energy data
            response_json = remove_dict_none_values(combine_chunks(chunks))
            energy_data = client.get_last_energy_data()
            if energy_data:
                response_json["energy"] = energy_data
            response.response_json = response_json
            
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.id,
                            name=value.function.name,
                            arguments=json.loads(value.function.arguments),
                        )
                    )
        else:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump() if completion.usage else None
            # Preserve ALL data including energy
            response_data = completion.model_dump()
            response.response_json = remove_dict_none_values(d=response_data)
            
            for tool_call in completion.choices[0].message.tool_calls or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, str)
                        else tool_call.function.arguments,
                    )
                )
            if completion.choices[0].message.content is not None:
                yield completion.choices[0].message.content
        
        self.set_usage(response, usage)
        response._prompt_json = {"messages": messages}


class NeuralWattAsyncChat(NeuralWattShared, llm.AsyncKeyModel):
    default_max_tokens = None

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        """
        Execute a chat completion sending a request to the NeuralWatt API,
        preserving returned energy data to store in llm's local logs.db.
        """
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key, async_=True)
        usage = None

        if stream:
            completion = await client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            tool_calls = {}

            async for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta:
                    for tool_call in chunk.choices[0].delta.tool_calls or []:
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ""
                        index = tool_call.index
                        if index not in tool_calls:
                            tool_calls[index] = tool_call
                        else:
                            tool_calls[
                                index
                            ].function.arguments += tool_call.function.arguments
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content

            # Combine chunks and add energy data
            response_json = remove_dict_none_values(combine_chunks(chunks))
            energy_data = client.get_last_energy_data()
            if energy_data:
                response_json["energy"] = energy_data
            response.response_json = response_json

            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.id,
                            name=value.function.name,
                            arguments=json.loads(value.function.arguments),
                        )
                    )
        else:
            completion = await client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump() if completion.usage else None
            # Preserve ALL data including energy
            response_data = completion.model_dump()
            response.response_json = remove_dict_none_values(response_data)

            for tool_call in completion.choices[0].message.tool_calls or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, str)
                        else tool_call.function.arguments,
                    )
                )
            if completion.choices[0].message.content is not None:
                yield completion.choices[0].message.content
        
        self.set_usage(response, usage)
        response._prompt_json = {"messages": messages}
