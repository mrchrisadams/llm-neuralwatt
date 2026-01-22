# Streaming Support for Energy Data in llm-neuralwatt

## Problem Statement

Currently, the llm-neuralwatt plugin cannot capture energy consumption data when streaming is enabled. This is because Neuralwatt sends energy data as an SSE comment (line starting with `:`) just before the `[DONE]` marker, and the OpenAI Python client library ignores SSE comments per the SSE specification.

### Current Streaming Response Format

When making a streaming request to Neuralwatt, the response looks like:

```
data: {"id": "chatcmpl-xxx", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hello"}, ...}]}
data: {"id": "chatcmpl-xxx", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "!"}, ...}]}
data: {"id": "chatcmpl-xxx", ..., "choices": [], "usage": {...}}
: energy {"energy_joules": 30.42, "energy_kwh": 8.45e-06, "avg_power_watts": 78.5, "duration_seconds": 0.388, "attribution_method": "prorated", "attribution_ratio": 1.0}
data: [DONE]
```

The key insight is that the energy data is sent as an **SSE comment** (the line starting with `: energy`).

### Why OpenAI Client Doesn't Work

The OpenAI client's SSE decoder in `_streaming.py` follows the SSE specification:

```python
def decode(self, line: str) -> ServerSentEvent | None:
    if line.startswith(":"):
        return None  # Comments are ignored
```

This is correct per the SSE spec, but it means we can't capture Neuralwatt's energy data.

## Solution Approach

### Custom HTTP-based Streaming

We will implement custom streaming using httpx directly instead of the OpenAI client. This allows us to:

1. **Parse SSE ourselves** - capturing both `data:` lines AND `: energy` comments
2. **Yield content in real-time** - maintaining the streaming user experience
3. **Capture energy data** - storing it when we encounter the energy comment
4. **Attach energy to response_json** - after streaming completes

### Implementation Details

#### 1. Custom SSE Parser

Create a generator function that:
- Iterates through the raw HTTP response bytes
- Parses SSE lines (splitting on `\n\n`)
- Yields content chunks from `data:` lines
- Captures and stores `energy` data from comment lines
- Returns energy data after iteration completes

#### 2. Modified execute() Method

In streaming mode:
- Use httpx to make the POST request with `stream=True`
- Use custom SSE parser to process the response
- Build messages list from chunks (same as current)
- After streaming, combine chunks AND energy data into response_json

#### 3. Async Support

Same approach for `NeuralWattAsyncChat`, using `httpx.AsyncClient`.

### Code Structure

```python
def _parse_sse_stream(response_iter):
    """Parse SSE stream, yielding (event_type, data) tuples.
    
    event_type is one of: 'chunk', 'energy', 'done'
    """
    buffer = ""
    for chunk in response_iter:
        buffer += chunk.decode('utf-8')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith(': energy '):
                # Energy comment
                energy_json = line[9:]  # Skip ": energy "
                yield ('energy', json.loads(energy_json))
            elif line.startswith('data: '):
                data = line[6:]  # Skip "data: "
                if data == '[DONE]':
                    yield ('done', None)
                else:
                    yield ('chunk', json.loads(data))


def execute(self, prompt, stream, response, conversation=None, key=None):
    if stream:
        # Use custom httpx-based streaming
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        
        with httpx.Client() as client:
            http_response = client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.get_key(key)}", ...},
                json={"model": self.model_name, "messages": messages, "stream": True, **kwargs},
                timeout=None
            )
            
            chunks = []
            energy_data = None
            
            for event_type, data in _parse_sse_stream(http_response.iter_bytes()):
                if event_type == 'chunk':
                    chunks.append(data)
                    # Extract and yield content
                    if data.get('choices') and data['choices'][0].get('delta', {}).get('content'):
                        yield data['choices'][0]['delta']['content']
                elif event_type == 'energy':
                    energy_data = data
                elif event_type == 'done':
                    break
        
        # Combine chunks into response_json, including energy
        combined = combine_chunks_from_dicts(chunks)
        if energy_data:
            combined['energy'] = energy_data
        response.response_json = combined
    else:
        # Existing non-streaming code...
```

## Benefits

1. **No external dependencies** - Uses httpx which is already a dependency of openai
2. **Maintains streaming UX** - Tokens appear in real-time as before
3. **Captures energy data** - Stored in response_json like non-streaming mode
4. **Self-contained** - No changes needed to LLM library or OpenAI client

## Implementation Status

✅ **COMPLETED**

### Changes Made

1. **`llm_neuralwatt.py`** - Implemented custom HTTP streaming:
   - Added `_parse_sse_line()` function to parse SSE lines including energy comments
   - Added `_combine_streaming_chunks()` to combine chunks with energy data
   - Modified `NeuralWattChat.execute()` to use httpx for streaming with custom SSE parsing
   - Modified `NeuralWattAsyncChat.execute()` to use httpx.AsyncClient for async streaming
   - Non-streaming mode unchanged (uses OpenAI client)

2. **`tests/test_streaming.py`** - Added comprehensive tests:
   - Tests for `_parse_sse_line()` covering energy comments, data chunks, [DONE], etc.
   - Tests for `_combine_streaming_chunks()` covering content concatenation, energy inclusion, metadata

3. **`README.md`** - Updated documentation:
   - Removed "Known issues with Streaming" section
   - Added "Streaming Support" section confirming both modes work

### Verification

Manual testing confirmed:
- ✅ Streaming mode captures energy data
- ✅ Non-streaming mode captures energy data  
- ✅ All unit tests pass (12/12)
- ✅ Energy data visible in `llm logs --json`
