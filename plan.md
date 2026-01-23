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

### Subclassing the OpenAI Client

The cleanest approach is to subclass the OpenAI client's SSE decoder and override the `decode()` method to capture energy comments before they're discarded.

#### Key Components

1. **`EnergyCapturingSSEDecoder`** - Subclass of `openai._streaming.SSEDecoder` that:
   - Checks for `: energy ` prefix before calling parent's decode()
   - Parses and stores energy JSON in `self.energy_data`
   - Passes all other lines to the parent decoder unchanged

2. **`NeuralWattOpenAI`** / **`NeuralWattAsyncOpenAI`** - Subclasses of OpenAI clients that:
   - Override `_make_sse_decoder()` to return `EnergyCapturingSSEDecoder`
   - Store reference to decoder to access energy data after streaming
   - Provide `get_last_energy_data()` method

3. **Modified `get_client()`** in `NeuralWattShared` to return the custom clients

### Benefits of This Approach

1. **Minimal code changes** - Only ~60 lines of new code
2. **Preserves all OpenAI client features** - Retries, timeouts, proxies, logging, etc.
3. **Type-safe** - Still get Pydantic models from the client
4. **Consistent** - Non-streaming and streaming use the same client infrastructure
5. **Maintainable** - Changes are isolated to the decoder layer

## Implementation Status

✅ **COMPLETED**

### Changes Made

1. **`llm_neuralwatt.py`**:
   - Added `EnergyCapturingSSEDecoder` class
   - Added `NeuralWattOpenAI` and `NeuralWattAsyncOpenAI` client subclasses
   - Modified `NeuralWattShared.get_client()` to return custom clients
   - Updated streaming code to call `client.get_last_energy_data()` after iteration

2. **`tests/test_streaming.py`**:
   - Tests for `EnergyCapturingSSEDecoder` covering energy capture, regular comments, data passthrough

3. **`README.md`**:
   - Removed "Known issues with Streaming" section
   - Added "Streaming Support" section confirming both modes work

### Verification

Manual testing confirmed:
- ✅ Streaming mode captures energy data
- ✅ Non-streaming mode captures energy data  
- ✅ All unit tests pass (7/7)
- ✅ Energy data visible in `llm logs --json`
- ✅ All OpenAI client features preserved (retries, headers, logging, etc.)
