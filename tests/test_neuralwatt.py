from llm.plugins import load_plugins, pm
import pytest
import json
from unittest.mock import patch, MagicMock


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_neuralwatt" in names


def test_sse_parsing():
    """Test the SSE parser for energy events"""
    from llm_neuralwatt import NeuralWattSSEParser
    
    parser = NeuralWattSSEParser()
    
    # Test energy parsing
    energy_line = ': energy {"energy_joules": 42.0, "energy_kwh": 0.00001}'
    parser.parse_lines([energy_line])
    
    energy = parser.get_last_energy()
    assert energy is not None
    assert energy["energy_joules"] == 42.0
    assert energy["energy_kwh"] == 0.00001


def test_stream_energy_capture():
    """Test that streaming responses capture energy data"""
    from llm_neuralwatt import NeuralWattDirectStream
    import httpx
    
    # Mock response with energy data
    mock_response = MagicMock()
    lines = [
        b'data: {"id": "test", "choices": [{"index": 0, "delta": {"content": "Hello"}}]}',
        b'data: {"id": "test", "choices": [{"index": 0, "delta": {"content": " world"}}]}',
        b': energy {"energy_joules": 50.0, "energy_kwh": 0.000015}',
        b'data: [DONE]'
    ]
    mock_response.iter_lines.return_value = lines
    
    stream = NeuralWattDirectStream(mock_response, "test-model", "https://api.test.com")
    
    # Iterate through stream
    chunks = list(stream)
    assert len(chunks) == 2
    
    # Check energy was captured
    assert stream.captured_energy is not None
    assert stream.captured_energy["energy_joules"] == 50.0
    
    # Check combined response includes energy
    response = stream.get_combined_response()
    assert "energy" in response
    assert response["energy"]["energy_joules"] == 50.0


def test_async_stream_energy_capture():
    """Test that async streaming responses capture energy data"""
    import asyncio
    from llm_neuralwatt import NeuralWattDirectAsyncStream
    
    async def test_async():
        # Mock response
        mock_response = MagicMock()
        lines = [
            b'data: {"id": "test", "choices": [{"index": 0, "delta": {"content": "Hi"}}]}',
            b': energy {"energy_joules": 25.0, "energy_kwh": 0.000007}',
            b'data: [DONE]'
        ]
        
        async def mock_aiter_lines():
            for line in lines:
                yield line
        
        mock_response.aiter_lines = mock_aiter_lines
        
        stream = NeuralWattDirectAsyncStream(mock_response, "test-model", "https://api.test.com")
        
        # Iterate through stream
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert stream.captured_energy is not None
        assert stream.captured_energy["energy_joules"] == 25.0
        
        # Check combined response
        response = await stream.get_combined_response()
        assert "energy" in response
        assert response["energy"]["energy_joules"] == 25.0
    
    # Run the async test
    asyncio.run(test_async())


def test_integration_streaming():
    """Integration test for streaming with energy capture"""
    # This test would require more complex mocking of the framework
    # For now, we'll just verify the components work
    pass
