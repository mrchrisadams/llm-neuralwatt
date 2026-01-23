# Let's take a much simpler approach - directly handle NeuralWatt SSE format

class NeuralWattSSEParser:
    """Parser for NeuralWatt's SSE format with energy events"""
    
    def __init__(self):
        self.energies = []
    
    def parse_lines(self, lines):
        """Parse SSE lines and extract energy data"""
        for line in lines:
            if line.startswith(': energy '):
                energy_json = line[len(': energy '):].strip()
                try:
                    energy_data = json.loads(energy_json)
                    self.energies.append(energy_data)
                except json.JSONDecodeError:
                    pass
    
    def get_last_energy(self):
        return self.energies[-1] if self.energies else None


import json
import httpx
from typing import Iterator, Optional, Union
from openai import Stream, AsyncStream, OpenAI
from llm.default_plugins.openai_models import remove_dict_none_values, combine_chunks
import re


class NeuralWattDirectStream:
    """A direct implementation of streaming that handles NeuralWatt's SSE format"""
    
    def __init__(self, response: httpx.Response, model_name: str, api_base: str):
        self.response = response
        self.model_name = model_name
        self.api_base = api_base
        self.captured_energy = None
        self._chunks = []
        
    def __iter__(self):
        # Parse the SSE stream directly
        buffer = ""
        parser = NeuralWattSSEParser()
        
        for line in self.response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode('utf-8', errors='ignore')
            
            if line.startswith(': energy '):
                # Parse energy data
                parser.parse_lines([line])
                energy_data = parser.get_last_energy()
                if energy_data:
                    self.captured_energy = energy_data
                continue
            
            if line.startswith('data: '):
                data_str = line[len('data: '):]
                if data_str == '[DONE]':
                    break
                
                try:
                    data = json.loads(data_str)
                    # Convert to the OpenAI chunk object
                    # Using a simple dict approach for now
                    self._chunks.append(data)
                    yield data
                except json.JSONDecodeError:
                    pass
        
        # Ensure we have the last energy data
        final_energy = parser.get_last_energy()
        if final_energy:
            self.captured_energy = final_energy
    
    def get_combined_response(self):
        """Get the final response with energy data"""
        if not self._chunks:
            return {}
        
        # Use the last chunk as the base and add energy
        response = self._chunks[-1].copy()
        if self.captured_energy:
            response['energy'] = self.captured_energy
        
        return response


class NeuralWattDirectAsyncStream:
    """Async version of NeuralWattDirectStream"""
    
    def __init__(self, response: httpx.Response, model_name: str, api_base: str):
        self.response = response
        self.model_name = model_name
        self.api_base = api_base
        self.captured_energy = None
        self._chunks = []
        self._parser = NeuralWattSSEParser()
        
    async def __aiter__(self):
        buffer = ""
        
        async for line in self.response.aiter_lines():
            if isinstance(line, bytes):
                line_str = line.decode('utf-8', errors='ignore')
            else:
                line_str = line
            
            if line_str.startswith(': energy '):
                self._parser.parse_lines([line_str])
                energy_data = self._parser.get_last_energy()
                if energy_data:
                    self.captured_energy = energy_data
                continue
            
            if line_str.startswith('data: '):
                data_str = line_str[len('data: '):]
                if data_str == '[DONE]':
                    break
                
                try:
                    data = json.loads(data_str)
                    self._chunks.append(data)
                    yield data
                except json.JSONDecodeError:
                    pass
        
        # Ensure final energy
        final_energy = self._parser.get_last_energy()
        if final_energy:
            self.captured_energy = final_energy
    
    async def get_combined_response(self):
        """Get the final response with energy data"""
        if not self._chunks:
            return {}
        
        response = self._chunks[-1].copy()
        if self.captured_energy:
            response['energy'] = self.captured_energy
        
        return response


class EnergyCapturingStream:
    """Simplified wrapper - for now just pass through"""
    def __init__(self, stream):
        self._stream = stream
        self.captured_energy = None
        self._captured_chunks = []
    
    def __iter__(self):
        for chunk in self._stream:
            self._captured_chunks.append(chunk)
            yield chunk
    
    def get_combined_response(self):
        # Fallback - just combine chunks
        return remove_dict_none_values(combine_chunks(self._captured_chunks))


class EnergyCapturingAsyncStream:
    """Simplified async wrapper - for now just pass through"""
    def __init__(self, stream):
        self._stream = stream
        self.captured_energy = None
        self._captured_chunks = []
    
    async def __aiter__(self):
        async for chunk in self._stream:
            self._captured_chunks.append(chunk)
            yield chunk
    
    async def get_combined_response(self):
        return remove_dict_none_values(combine_chunks(self._captured_chunks))


class EnergyCapturingAsyncStream:
    """Async stream wrapper that captures energy data from NeuralWatt's custom SSE format"""
    
    def __init__(self, original_stream: AsyncStream):
        self._original_stream = original_stream
        self.captured_energy = None
        self._captured_chunks = []
        self._extracted_energy = False
        
    async def __aiter__(self):
        # For async, we have less ability to extract energy beforehand
        # We'll collect chunks and try to extract from the response later
        async for chunk in self._original_stream:
            self._captured_chunks.append(chunk)
            yield chunk
        
        # After stream is consumed, try to extract energy if needed
        if not self._extracted_energy:
            await self._extract_energy_data()
            self._extracted_energy = True
    
    async def _extract_energy_data(self):
        """Extract energy from async response"""
        try:
            response = self._original_stream.response
            if hasattr(response, '_content'):
                content = response._content.decode('utf-8', errors='ignore')
                self._parse_energy_from_content(content)
            elif hasattr(response, 'content'):
                content = response.content.decode('utf-8', errors='ignore')
                self._parse_energy_from_content(content)
            # Alternative: try to access through the HTTP client if available
            elif hasattr(self._original_stream, '_client') and hasattr(self._original_stream._client, '_history'):
                # Try to find the response in client history
                for req, resp in self._original_stream._client._history:
                    if hasattr(resp, 'content'):
                        content = resp.content.decode('utf-8', errors='ignore')
                        self._parse_energy_from_content(content)
                        break
        except Exception:
            pass  # Energy extraction failed, continue without it
    
    def _parse_energy_from_content(self, content: str):
        """Parse energy data from SSE content"""
        for line in content.split('\n'):
            if line.startswith(': energy '):
                energy_json = line[len(': energy '):].strip()
                try:
                    self.captured_energy = json.loads(energy_json)
                except json.JSONDecodeError:
                    pass
    
    async def get_combined_response(self):
        """Get combined response JSON with energy data"""
        response_json = remove_dict_none_values(combine_chunks(self._captured_chunks))
        
        if self.captured_energy and isinstance(response_json, dict):
            response_json['energy'] = self.captured_energy
        
        return response_json


class NeuralWattHTTPClient(httpx.Client):
    """Custom HTTP client that can capture NeuralWatt's energy events"""
    
    def stream(self, method, url, **kwargs):
        """Override stream method to capture energy events"""
        response = super().stream(method, url, **kwargs)
        return response


class NeuralWattAsyncHTTPClient(httpx.AsyncClient):
    """Custom async HTTP client for NeuralWatt"""
    
    async def stream(self, method, url, **kwargs):
        """Override async stream method"""
        response = await super().astream(method, url, **kwargs)
        return response


# Energy extraction helper
def extract_energy_from_stream(response):
    """Extract energy data from HTTP response content"""
    try:
        if hasattr(response, 'content'):
            content = response.content.decode('utf-8', errors='ignore')
            for line in content.split('\n'):
                if line.startswith(': energy '):
                    energy_json = line[len(': energy '):].strip()
                    return json.loads(energy_json)
    except Exception:
        pass
    return None


import llm
from llm.default_plugins.openai_models import _Shared
from llm.default_plugins.openai_models import (
    remove_dict_none_values,
    combine_chunks,
)
import json


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


class NeuralWattChat(NeuralWattShared, llm.KeyModel):
    default_max_tokens = None

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """
        Execute a chat completion sending a request to the NeuralWatt API,
        preserving returned energy data to store in llm's local logs.db
        Only non-streaming is currently supported, because the final 'energy' chunk
        is only available at the end of the response, and it looks like it gets
        filtered out by the OpenAI client's streaming implementation.
        """
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None

        if stream:
            # For streaming, use direct HTTP request to capture energy data
            import httpx
            
            # Create the request payload
            payload = {
                "model": self.model_name or self.model_id,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            
            # Make direct HTTP request to capture raw SSE stream  
            with httpx.Client() as http_client:
                with http_client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as http_response:
                    # Create direct stream
                    direct_stream = NeuralWattDirectStream(
                        http_response, 
                        self.model_name or self.model_id,
                        self.api_base
                    )
                    
                    # Process chunks
                    chunks = []
                    for chunk_data in direct_stream:
                        # Convert dict to proper ChatCompletionChunk-like object
                        # For now, just extract content
                        if 'choices' in chunk_data and chunk_data['choices']:
                            choice = chunk_data['choices'][0]  
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                if content is not None:
                                    yield content
                        chunks.append(chunk_data)
                    
                    # Get the final response with energy data
                    response.response_json = direct_stream.get_combined_response()
                    
                    # Extract usage data
                    if 'usage' in response.response_json:
                        usage = response.response_json['usage']
                    else:
                        usage = None
        else:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump() if completion.usage else None
            # Preserve ALL data including energy - don't filter unnecessarily early
            response_data = completion.model_dump()

            # Apply remove_dict_none_values only at the end
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
        preserving returned energy data to store in llm's local logs.db
        Only non-streaming is currently supported, because the final 'energy' chunk
        is only available at the end of the response, and it looks like it gets
        filtered out by the OpenAI client's streaming implementation.
        """
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key, async_=True)
        usage = None

        if stream:
            # For async streaming, use direct HTTP request
            import httpx
            
            # Create the request payload  
            payload = {
                "model": self.model_name or self.model_id,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            
            # Make direct async HTTP request
            async with httpx.AsyncClient() as http_client:
                async with http_client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as http_response:
                    # Create async direct stream
                    direct_stream = NeuralWattDirectAsyncStream(
                        http_response,
                        self.model_name or self.model_id,
                        self.api_base
                    )
                    
                    # Process chunks asynchronously
                    chunks = []
                    async for chunk_data in direct_stream:
                        if 'choices' in chunk_data and chunk_data['choices']:
                            choice = chunk_data['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                if content is not None:
                                    yield content
                        chunks.append(chunk_data)
                    
                    # Get final response with energy data
                    response.response_json = await direct_stream.get_combined_response()
                    
                    # Extract usage data
                    if 'usage' in response.response_json:
                        usage = response.response_json['usage']
                    else:
                        usage = None
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

            # Apply remove_dict_none_values only at the end
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
