"""Tests for streaming energy data capture in llm-neuralwatt."""
import json
import pytest
from llm_neuralwatt import EnergyCapturingSSEDecoder


class TestEnergyCapturingSSEDecoder:
    """Test the custom SSE decoder that captures energy data."""

    def test_captures_energy_comment(self):
        """Energy comments should be captured and stored."""
        decoder = EnergyCapturingSSEDecoder()
        line = ': energy {"energy_joules": 15.23, "energy_kwh": 0.00000423}'
        result = decoder.decode(line)
        
        assert result is None  # Should not emit an event
        assert decoder.energy_data is not None
        assert decoder.energy_data['energy_joules'] == 15.23
        assert decoder.energy_data['energy_kwh'] == 0.00000423

    def test_ignores_regular_comments(self):
        """Regular SSE comments should be ignored (not stored as energy)."""
        decoder = EnergyCapturingSSEDecoder()
        result = decoder.decode(': this is a regular comment')
        
        assert result is None
        assert decoder.energy_data is None

    def test_passes_data_lines_to_parent(self):
        """Standard data lines should be handled by parent decoder."""
        decoder = EnergyCapturingSSEDecoder()
        
        # First decode an empty line to flush the buffer (SSE spec)
        decoder.decode('data: {"test": "value"}')
        result = decoder.decode('')
        
        assert result is not None
        assert result.data == '{"test": "value"}'

    def test_handles_malformed_energy_json(self):
        """Malformed energy JSON should not crash."""
        decoder = EnergyCapturingSSEDecoder()
        result = decoder.decode(': energy {not valid json}')
        
        assert result is None
        assert decoder.energy_data is None  # Should not be set

    def test_energy_data_initially_none(self):
        """Energy data should be None before any energy comment is seen."""
        decoder = EnergyCapturingSSEDecoder()
        assert decoder.energy_data is None

    def test_full_energy_payload(self):
        """Test with a realistic energy payload from Neuralwatt."""
        decoder = EnergyCapturingSSEDecoder()
        line = ': energy {"energy_joules": 30.42, "energy_kwh": 8.45e-06, "avg_power_watts": 78.5, "duration_seconds": 0.388, "attribution_method": "prorated", "attribution_ratio": 1.0}'
        decoder.decode(line)
        
        assert decoder.energy_data['energy_joules'] == 30.42
        assert decoder.energy_data['avg_power_watts'] == 78.5
        assert decoder.energy_data['attribution_method'] == 'prorated'
