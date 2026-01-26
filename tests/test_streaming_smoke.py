"""Additional smoke test for streaming functionality in llm-neuralwatt plugin."""

import subprocess
import sys
import pytest
import os


def test_llm_neuralwatt_streaming_smoke_test():
    """Smoke test for streaming API functionality."""

    # Get the API key from the file
    with open("neuralwatt.api.key.txt", "r") as f:
        api_key = f.read().strip()

    # Ensure the key is set in environment variable
    env = os.environ.copy()
    env["NEURALWATT_API_KEY"] = api_key

    # Test streaming API with a simple prompt
    try:
        # Using a simple prompt to test streaming
        cmd = [
            sys.executable,
            "-m",
            "llm",
            "prompt",
            "Say hello",
            "-m",
            "neuralwatt-gpt-oss",
            "--no-log",  # Don't log to avoid database issues in tests
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=env
        )

        # Check that the command executed successfully
        assert result.returncode == 0, (
            f"LLM streaming command failed with return code {result.returncode}: {result.stderr}"
        )

        # Check that we got a response
        response_text = result.stdout.strip()
        assert len(response_text) > 0, "No response received from streaming API"

        # Verify the response contains expected content
        assert "hello" in response_text.lower() or "Hello" in response_text, (
            "Response doesn't seem to contain expected content"
        )

    except subprocess.TimeoutExpired:
        pytest.fail(
            "LLM streaming command timed out - NeuralWatt API may be unreachable"
        )
    except Exception as e:
        pytest.fail(f"Unexpected error during streaming smoke test: {e}")


def test_llm_neuralwatt_streaming_vs_non_streaming():
    """Compare streaming vs non-streaming responses."""

    # Get the API key from the file
    with open("neuralwatt.api.key.txt", "r") as f:
        api_key = f.read().strip()

    # Ensure the key is set in environment variable
    env = os.environ.copy()
    env["NEURALWATT_API_KEY"] = api_key

    try:
        # Test with streaming (default)
        streaming_cmd = [
            sys.executable,
            "-m",
            "llm",
            "prompt",
            "What is the capital of France?",
            "-m",
            "neuralwatt-gpt-oss",
            "--no-log",
        ]

        streaming_result = subprocess.run(
            streaming_cmd, capture_output=True, text=True, timeout=30, env=env
        )

        # Check that the streaming command executed successfully
        assert streaming_result.returncode == 0, (
            f"Streaming command failed with return code {streaming_result.returncode}: {streaming_result.stderr}"
        )

        # Test with explicit non-streaming
        non_streaming_cmd = [
            sys.executable,
            "-m",
            "llm",
            "prompt",
            "What is the capital of France?",
            "-m",
            "neuralwatt-gpt-oss",
            "--no-stream",
            "--no-log",
        ]

        non_streaming_result = subprocess.run(
            non_streaming_cmd, capture_output=True, text=True, timeout=30, env=env
        )

        # Check that the non-streaming command executed successfully
        assert non_streaming_result.returncode == 0, (
            f"Non-streaming command failed with return code {non_streaming_result.returncode}: {non_streaming_result.stderr}"
        )

        # Both should return responses
        streaming_response = streaming_result.stdout.strip()
        non_streaming_response = non_streaming_result.stdout.strip()

        assert len(streaming_response) > 0, "No response from streaming API"
        assert len(non_streaming_response) > 0, "No response from non-streaming API"

        # Both responses should contain the expected answer
        assert "Paris" in streaming_response or "paris" in streaming_response.lower(), (
            "Streaming response doesn't mention Paris"
        )
        assert (
            "Paris" in non_streaming_response
            or "paris" in non_streaming_response.lower()
        ), "Non-streaming response doesn't mention Paris"

    except subprocess.TimeoutExpired:
        pytest.fail("LLM command timed out - NeuralWatt API may be unreachable")
    except Exception as e:
        pytest.fail(f"Unexpected error during streaming comparison test: {e}")
