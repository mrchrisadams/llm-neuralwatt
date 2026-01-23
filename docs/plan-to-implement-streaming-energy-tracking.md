# Plan: Adding Streaming Support to llm-neuralwatt

## Problem Statement

Currently, the llm-neuralwatt plugin only captures energy consumption data when using non-streaming requests. When streaming, energy data chunks are filtered out by the OpenAI client library.

## Analysis

### Current Behavior
- **Non-streaming (works)**: Energy data is captured and stored in `response_json.energy`
- **Streaming (broken)**: Energy data chunks are lost during streaming

### Root Cause
The OpenAI client library's streaming implementation filters out non-standard chunks. NeuralWatt sends energy data as a special chunk just before the `[DONE]` marker, but this gets filtered.

### Key Files Examined
- `/home/exedev/llm-neuralwatt/llm_neuralwatt.py` - Main plugin implementation
- `/home/exedev/llm-neuralwatt/venv/lib/python3.12/site-packages/openai/_streaming.py` - OpenAI streaming implementation
- Current plugin extends `llm.default_plugins.openai_models._Shared`

## Solution Approach

We need to create a custom streaming implementation that:

1. **Subclass the OpenAI Stream classes** to capture raw SSE events
2. **Extract energy chunks** before they're filtered by the standard processing
3. **Preserve energy data** alongside regular content chunks
4. **Integrate seamlessly** with the existing LLM framework

### Implementation Strategy

#### Option 1: Custom Stream Classes
Create custom stream classes that inherit from OpenAI's Stream/AsyncStream but override the iteration logic to capture energy chunks.

#### Option 2: Custom SSE Decoder
Create a custom SSE decoder that can detect and preserve energy chunks.

#### Option 3: Client-level Hooking
Subclass the OpenAI client to intercept streaming responses at the HTTP level.

### Chosen Approach: Option 1 (Custom Stream Classes)

This approach:
- Maintains compatibility with the OpenAI client interface
- Only affects streaming paths
- Allows us to preserve all existing functionality
- Enables clean separation of concerns

## Detailed Implementation Plan

### Step 1: Create Custom Stream Classes
```python
class NeuralWattStream(openai.Stream):
    """Custom Stream class that captures energy chunks"""
    
class NeuralWattAsyncStream(openai.AsyncStream):
    """Custom AsyncStream class that captures energy chunks"""
```

### Step 2: Create Custom Client Factory
Add methods to create NeuralWatt-specific clients that return our custom stream classes.

### Step 3: Modify NeuralWattChat/AsyncChat
Update the execute methods to use the custom clients and store energy data from streaming responses.

### Step 4: Energy Data Storage
Create mechanism to store captured energy data in the response object alongside content.

### Step 5: Testing
- Verify non-streaming still works
- Verify streaming now captures energy
- Test async streaming
- Verify backward compatibility

## Code Structure Changes

### New Classes to Add
1. `NeuralWattStream` - Custom sync stream with energy capture
2. `NeuralWattAsyncStream` - Custom async stream with energy capture  
3. `NeuralWattSSEDecoder` - Custom decoder to detect energy chunks
4. `NeuralWattStreamClient` - Client factory returning custom streams

### Files to Modify
1. `llm_neuralwatt.py` - Main implementation
2. Tests for streaming functionality

## Backward Compatibility
- All existing functionality preserved
- No changes to public APIs
- Transparent to end users

## Testing Strategy
1. Unit tests for new classes
2. Integration tests with real API
3. Verify energy data captured in streaming mode
4. Performance impact assessment
