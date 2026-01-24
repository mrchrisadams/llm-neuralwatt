# Architecture: Energy Data Collection in llm-neuralwatt

This document explains how the llm-neuralwatt plugin captures energy consumption data from the Neuralwatt API, including both streaming and non-streaming modes.

## Overview

Neuralwatt is an OpenAI-compatible inference API that measures and reports the energy consumption of each request. This plugin integrates with Simon Willison's [LLM](https://llm.datasette.io/) tool to make Neuralwatt models available and persist energy data in LLM's local SQLite database.

## C4 Model Diagrams

### Context Diagram

Shows the system's relationship with users and external systems.

![c4-context](./llm-neuralwatt-c4-context.png)


### Container Diagram

Shows the high-level components and how they interact.

![c4-context](./llm-neuralwatt-c4-containers.png)

### Component Diagram

Shows the internal components of the llm-neuralwatt plugin.


![c4-context](./llm-neuralwatt-c4-components.png)

### Class Diagram

Shows the inheritance hierarchy for energy capture.

```mermaid
classDiagram
    class SSEDecoder {
        <<openai._streaming>>
        +decode(line: str) ServerSentEvent|None
        +iter_bytes(iterator) Iterator
        +aiter_bytes(iterator) AsyncIterator
    }

    class EnergyCapturingSSEDecoder {
        +energy_data: dict|None
        +decode(line: str) ServerSentEvent|None
    }

    class OpenAI {
        <<openai>>
        +_make_sse_decoder() SSEDecoder
        +chat: ChatCompletions
    }

    class NeuralWattOpenAI {
        -_last_decoder: EnergyCapturingSSEDecoder
        +_make_sse_decoder() EnergyCapturingSSEDecoder
        +get_last_energy_data() dict|None
    }

    class AsyncOpenAI {
        <<openai>>
        +_make_sse_decoder() SSEDecoder
        +chat: AsyncChatCompletions
    }

    class NeuralWattAsyncOpenAI {
        -_last_decoder: EnergyCapturingSSEDecoder
        +_make_sse_decoder() EnergyCapturingSSEDecoder
        +get_last_energy_data() dict|None
    }

    SSEDecoder <|-- EnergyCapturingSSEDecoder : extends
    OpenAI <|-- NeuralWattOpenAI : extends
    AsyncOpenAI <|-- NeuralWattAsyncOpenAI : extends
    NeuralWattOpenAI ..> EnergyCapturingSSEDecoder : creates
    NeuralWattAsyncOpenAI ..> EnergyCapturingSSEDecoder : creates
```

## How Energy Data is Transmitted

### Non-Streaming Mode

In non-streaming mode, Neuralwatt includes energy data directly in the JSON response:

```json
{
  "id": "chatcmpl-abc123",
  "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5},
  "energy": {
    "energy_joules": 15.23,
    "energy_kwh": 4.23e-06,
    "avg_power_watts": 78.5,
    "duration_seconds": 0.194,
    "attribution_method": "prorated",
    "attribution_ratio": 1.0
  }
}
```

The standard OpenAI client parses this correctly, and LLM stores it in `response_json`.

### Streaming Mode

In streaming mode, Neuralwatt uses Server-Sent Events (SSE). The challenge is that energy data arrives as an **SSE comment** just before `[DONE]`:

```
data: {"choices": [{"delta": {"content": "Hello"}}]}
data: {"choices": [{"delta": {"content": "!"}}]}
data: {"choices": [], "usage": {...}}
: energy {"energy_joules": 15.23, "energy_kwh": 4.23e-06, ...}
data: [DONE]
```

Per the [SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html), lines starting with `:` are comments and should be ignored. The standard OpenAI client follows this spec:

```python
# From openai/_streaming.py
def decode(self, line: str) -> ServerSentEvent | None:
    if line.startswith(":"):
        return None  # Comments are ignored
```

#### What enables the logging in streaming mode: A custom SSE Decoder

We subclass the SSE decoder to capture energy comments before they're discarded. 

## Data Flow: Streaming Request

```mermaid
sequenceDiagram
    participant User
    participant LLM as LLM Tool
    participant Chat as NeuralWattChat
    participant Client as NeuralWattOpenAI
    participant Decoder as EnergyCapturingSSEDecoder
    participant API as Neuralwatt API

    User->>LLM: llm "Hello" -m neuralwatt-gpt-oss
    LLM->>Chat: execute(prompt, stream=True)
    Chat->>Client: get_client()
    Chat->>Client: chat.completions.create(stream=True)
    Client->>API: POST /v1/chat/completions
    
    loop SSE Stream
        API-->>Client: data: {"delta": {"content": "Hi"}}
        Client->>Decoder: decode(line)
        Decoder-->>Client: ServerSentEvent
        Client-->>Chat: chunk
        Chat-->>LLM: yield "Hi"
        LLM-->>User: "Hi"
    end
    
    API-->>Client: : energy {"energy_joules": 15.23, ...}
    Client->>Decoder: decode(line)
    Note over Decoder: Captures energy data<br/>stores in self.energy_data
    Decoder-->>Client: None (not emitted as event)
    
    API-->>Client: data: [DONE]
    
    Chat->>Client: get_last_energy_data()
    Client-->>Chat: {"energy_joules": 15.23, ...}
    Chat->>Chat: Add energy to response_json
    Chat-->>LLM: response with energy
    LLM->>LLM: Store in logs.db
```

## Energy Data Fields

Each energy measurement from Neuralwatt includes:

| Field | Type | Description |
|-------|------|-------------|
| `energy_joules` | float | Total energy consumed in joules |
| `energy_kwh` | float | Total energy consumed in kilowatt-hours |
| `avg_power_watts` | float | Average power draw during inference |
| `duration_seconds` | float | Time taken for the request |
| `attribution_method` | string | How energy was attributed (e.g., "prorated") |
| `attribution_ratio` | float | Ratio of total GPU energy attributed to this request |

For more details on attribution methodology, see the [Neuralwatt documentation](https://portal.neuralwatt.com/docs/energy-methodology).

## Other approaches considered

We considered replacing the OpenAI client with raw `httpx` requests, but subclassing preserves:

- **Automatic retries** with exponential backoff
- **Timeout handling** with sensible defaults
- **Proxy support** via environment variables
- **Debug logging** via `LLM_OPENAI_SHOW_RESPONSES`
- **Custom headers** support
- **Type safety** with Pydantic response models

The subclass approach requires only ~60 lines of code vs ~200 for a full replacement.
