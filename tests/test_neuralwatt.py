from llm.plugins import load_plugins, pm
import pytest
import json
from unittest.mock import patch, MagicMock
from llm_neuralwatt import EnergyCapturingSSEDecoder


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_neuralwatt" in names


def test_energy_capturing_sse_decoder():
    """Test the EnergyCapturingSSEDecoder for energy events"""
    decoder = EnergyCapturingSSEDecoder()
    
    # Test energy comment parsing
    event = decoder.decode(": energy {\"energy_joules\": 42.0, \"energy_kwh\": 0.00001}")
    
    # Energy events should not be emitted as events, but captured
    assert event is None
    assert decoder.energy_data is not None
    assert decoder.energy_data["energy_joules"] == 42.0
    assert decoder.energy_data["energy_kwh"] == 0.00001
    
    # Test standard SSE lines still work
    event = decoder.decode("data: {\"id\": \"test\"}")
    assert event is not None
    assert event.data == "{\"id\": \"test\"}"
    assert event.event is None


def test_neuralwatt_openai_client():
    """Test NeuralWattOpenAI client captures energy data"""
    from llm_neuralwatt import NeuralWattOpenAI
    
    client = NeuralWattOpenAI(api_key="test", base_url="https://api.neuralwatt.com/v1")
    
    # Create decoder
    decoder = client._make_sse_decoder()
    assert isinstance(decoder, EnergyCapturingSSEDecoder)
    
    # Parse some SSE content
    decoder.decode("data: {\"id\": \"test\"}")
    decoder.decode(": energy {\"energy_joules\": 100.0, \"energy_kwh\": 0.00003}")
    decoder.decode("data: [DONE]")
    
    # Check energy was captured
    energy = client.get_last_energy_data()
    assert energy is not None
    assert energy["energy_joules"] == 100.0
    assert energy["energy_kwh"] == 0.00003


def test_neuralwatt_async_openai_client():
    """Test NeuralWattAsyncOpenAI client captures energy data"""
    from llm_neuralwatt import NeuralWattAsyncOpenAI
    import asyncio
    
    async def test_async():
        client = NeuralWattAsyncOpenAI(api_key="test", base_url="https://api.neuralwatt.com/v1")
        
        # Create decoder
        decoder = client._make_sse_decoder()
        assert isinstance(decoder, EnergyCapturingSSEDecoder)
        
        # Parse SSE content
        decoder.decode(": energy {\"energy_joules\": 200.0, \"energy_kwh\": 0.00006}")
        
        # Check energy was captured
        energy = client.get_last_energy_data()
        assert energy is not None
        assert energy["energy_joules"] == 200.0
    
    asyncio.run(test_async())


def test_sse_decoder_with_malformed_energy():
    """Test SSE decoder handles malformed energy events gracefully"""
    decoder = EnergyCapturingSSEDecoder()
    
    # Test malformed JSON
    event = decoder.decode(": energy not json")
    assert event is None
    assert decoder.energy_data is None
    
    # Test normal flow continues
    event = decoder.decode("data: {\"id\": \"test\"}")
    assert event is not None
