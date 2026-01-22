"""Tests for streaming energy data capture in llm-neuralwatt."""
import json
import pytest
from llm_neuralwatt import (
    _parse_sse_line,
    _combine_streaming_chunks,
)


class TestParseSSELine:
    """Test SSE line parsing."""

    def test_parse_energy_comment(self):
        """Energy comments should be parsed correctly."""
        line = ': energy {"energy_joules": 15.23, "energy_kwh": 0.00000423}'
        event_type, data = _parse_sse_line(line)
        assert event_type == 'energy'
        assert data['energy_joules'] == 15.23
        assert data['energy_kwh'] == 0.00000423

    def test_parse_regular_comment(self):
        """Regular SSE comments should be recognized but ignored."""
        line = ': this is a comment'
        event_type, data = _parse_sse_line(line)
        assert event_type == 'comment'

    def test_parse_data_chunk(self):
        """Standard data chunks should be parsed."""
        line = 'data: {"id": "test", "choices": [{"delta": {"content": "Hello"}}]}'
        event_type, data = _parse_sse_line(line)
        assert event_type == 'chunk'
        assert data['id'] == 'test'
        assert data['choices'][0]['delta']['content'] == 'Hello'

    def test_parse_done(self):
        """[DONE] marker should be recognized."""
        line = 'data: [DONE]'
        event_type, data = _parse_sse_line(line)
        assert event_type == 'done'
        assert data is None

    def test_parse_empty_line(self):
        """Empty lines should return None."""
        event_type, data = _parse_sse_line('')
        assert event_type is None
        assert data is None

    def test_parse_whitespace_line(self):
        """Whitespace-only lines should return None."""
        event_type, data = _parse_sse_line('   \t  ')
        assert event_type is None
        assert data is None


class TestCombineStreamingChunks:
    """Test combining streaming chunks."""

    def test_combine_content(self):
        """Content from multiple chunks should be concatenated."""
        chunks = [
            {'choices': [{'delta': {'role': 'assistant', 'content': 'Hello'}}]},
            {'choices': [{'delta': {'content': ' '}}]},
            {'choices': [{'delta': {'content': 'World'}}]},
        ]
        result = _combine_streaming_chunks(chunks)
        assert result['content'] == 'Hello World'
        assert result['role'] == 'assistant'

    def test_combine_with_energy(self):
        """Energy data should be included in combined result."""
        chunks = [
            {'choices': [{'delta': {'content': 'Hi'}}]},
        ]
        energy_data = {'energy_joules': 10.5, 'energy_kwh': 0.00001}
        result = _combine_streaming_chunks(chunks, energy_data)
        assert result['energy'] == energy_data

    def test_combine_preserves_metadata(self):
        """Metadata like id, model, created should be preserved."""
        chunks = [
            {
                'id': 'chatcmpl-123',
                'model': 'test-model',
                'created': 1234567890,
                'choices': [{'delta': {'content': 'test'}}],
            },
        ]
        result = _combine_streaming_chunks(chunks)
        assert result['id'] == 'chatcmpl-123'
        assert result['model'] == 'test-model'
        assert result['created'] == 1234567890

    def test_combine_extracts_usage(self):
        """Usage data should be extracted from chunks."""
        chunks = [
            {'choices': [{'delta': {'content': 'test'}}]},
            {
                'choices': [],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15,
                },
            },
        ]
        result = _combine_streaming_chunks(chunks)
        assert result['usage']['prompt_tokens'] == 10
        assert result['usage']['completion_tokens'] == 5

    def test_combine_extracts_finish_reason(self):
        """Finish reason should be extracted."""
        chunks = [
            {'choices': [{'delta': {'content': 'test'}}]},
            {'choices': [{'finish_reason': 'stop'}]},
        ]
        result = _combine_streaming_chunks(chunks)
        assert result['finish_reason'] == 'stop'
