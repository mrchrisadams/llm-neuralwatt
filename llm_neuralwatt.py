import llm
from llm.default_plugins.openai_models import _Shared
from llm.default_plugins.openai_models import (
    remove_dict_none_values,
    combine_chunks,
)
import httpx
import json


def _parse_sse_line(line):
    """
    Parse a single SSE line and return (event_type, data).
    
    event_type is one of: 'chunk', 'energy', 'done', 'comment', None
    """
    line = line.strip()
    if not line:
        return (None, None)
    
    # Check for energy comment (Neuralwatt-specific)
    if line.startswith(": energy "):
        energy_json = line[9:]  # Skip ": energy "
        try:
            return ('energy', json.loads(energy_json))
        except json.JSONDecodeError:
            return ('comment', line)
    
    # Standard SSE comment (starts with :)
    if line.startswith(":"):
        return ('comment', line)
    
    # Standard SSE data line
    if line.startswith("data: "):
        data = line[6:]  # Skip "data: "
        if data == "[DONE]":
            return ('done', None)
        try:
            return ('chunk', json.loads(data))
        except json.JSONDecodeError:
            return (None, None)
    
    return (None, None)


def _combine_streaming_chunks(chunks, energy_data=None):
    """
    Combine streaming chunks into a response dict, including energy data.
    
    This is similar to combine_chunks but works with raw dicts and includes energy.
    """
    content = ""
    role = None
    finish_reason = None
    usage = {}
    model = None
    response_id = None
    created = None
    
    for chunk in chunks:
        if chunk.get('usage'):
            usage = chunk['usage']
        if chunk.get('model'):
            model = chunk['model']
        if chunk.get('id'):
            response_id = chunk['id']
        if chunk.get('created'):
            created = chunk['created']
        
        choices = chunk.get('choices', [])
        for choice in choices:
            delta = choice.get('delta', {})
            if delta.get('role'):
                role = delta['role']
            if delta.get('content') is not None:
                content += delta['content']
            if choice.get('finish_reason'):
                finish_reason = choice['finish_reason']
    
    combined = {
        "content": content,
        "role": role,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    
    if response_id:
        combined["id"] = response_id
    if model:
        combined["model"] = model
    if created:
        combined["created"] = created
    if energy_data:
        combined["energy"] = energy_data
    
    return remove_dict_none_values(combined)


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
        preserving returned energy data to store in llm's local logs.db.
        
        For streaming requests, we use httpx directly to capture the energy
        data that Neuralwatt sends as an SSE comment before [DONE].
        """
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        usage = None

        if stream:
            # Use custom httpx-based streaming to capture energy data
            api_key = self.get_key(key)
            
            request_body = {
                "model": self.model_name or self.model_id,
                "messages": messages,
                "stream": True,
                **kwargs,
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            chunks = []
            energy_data = None
            tool_calls = {}
            
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=request_body,
                ) as http_response:
                    http_response.raise_for_status()
                    
                    buffer = ""
                    for chunk_bytes in http_response.iter_bytes():
                        buffer += chunk_bytes.decode('utf-8')
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            event_type, data = _parse_sse_line(line)
                            
                            if event_type == 'energy':
                                energy_data = data
                            elif event_type == 'chunk':
                                chunks.append(data)
                                
                                # Extract usage if present
                                if data.get('usage'):
                                    usage = data['usage']
                                
                                # Handle tool calls
                                choices = data.get('choices', [])
                                if choices and choices[0].get('delta'):
                                    delta = choices[0]['delta']
                                    for tc in delta.get('tool_calls', []) or []:
                                        if tc.get('function', {}).get('arguments') is None:
                                            tc['function']['arguments'] = ""
                                        index = tc.get('index', 0)
                                        if index not in tool_calls:
                                            tool_calls[index] = tc
                                        else:
                                            tool_calls[index]['function']['arguments'] += tc['function']['arguments']
                                
                                # Yield content
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content')
                                    if content is not None:
                                        yield content
                            elif event_type == 'done':
                                break
            
            # Combine chunks into response_json with energy data
            response.response_json = _combine_streaming_chunks(chunks, energy_data)
            
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.get('id'),
                            name=value.get('function', {}).get('name'),
                            arguments=json.loads(value.get('function', {}).get('arguments', '{}')),
                        )
                    )
        else:
            # Non-streaming: use OpenAI client (energy data is in the response)
            client = self.get_client(key)
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
        
        For streaming requests, we use httpx directly to capture the energy
        data that Neuralwatt sends as an SSE comment before [DONE].
        """
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        usage = None

        if stream:
            # Use custom httpx-based streaming to capture energy data
            api_key = self.get_key(key)
            
            request_body = {
                "model": self.model_name or self.model_id,
                "messages": messages,
                "stream": True,
                **kwargs,
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            chunks = []
            energy_data = None
            tool_calls = {}
            
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=request_body,
                ) as http_response:
                    http_response.raise_for_status()
                    
                    buffer = ""
                    async for chunk_bytes in http_response.aiter_bytes():
                        buffer += chunk_bytes.decode('utf-8')
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            event_type, data = _parse_sse_line(line)
                            
                            if event_type == 'energy':
                                energy_data = data
                            elif event_type == 'chunk':
                                chunks.append(data)
                                
                                # Extract usage if present
                                if data.get('usage'):
                                    usage = data['usage']
                                
                                # Handle tool calls
                                choices = data.get('choices', [])
                                if choices and choices[0].get('delta'):
                                    delta = choices[0]['delta']
                                    for tc in delta.get('tool_calls', []) or []:
                                        if tc.get('function', {}).get('arguments') is None:
                                            tc['function']['arguments'] = ""
                                        index = tc.get('index', 0)
                                        if index not in tool_calls:
                                            tool_calls[index] = tc
                                        else:
                                            tool_calls[index]['function']['arguments'] += tc['function']['arguments']
                                
                                # Yield content
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content')
                                    if content is not None:
                                        yield content
                            elif event_type == 'done':
                                break
            
            # Combine chunks into response_json with energy data
            response.response_json = _combine_streaming_chunks(chunks, energy_data)
            
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.get('id'),
                            name=value.get('function', {}).get('name'),
                            arguments=json.loads(value.get('function', {}).get('arguments', '{}')),
                        )
                    )
        else:
            # Non-streaming: use OpenAI client (energy data is in the response)
            client = self.get_client(key, async_=True)
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
