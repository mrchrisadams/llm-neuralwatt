"""Smoke tests for llm-neuralwatt plugin."""

import subprocess
import sys
import pytest
import os


def _get_api_key():
    """Get the API key from the file."""
    with open("neuralwatt.api.key.txt", "r") as f:
        return f.read().strip()


def _run_llm_command(cmd_args, timeout=30, use_api_key=True):
    """Helper to run LLM commands with proper environment setup."""
    env = os.environ.copy()
    if use_api_key:
        env["NEURALWATT_API_KEY"] = _get_api_key()

    cmd = [sys.executable, "-m", "llm", "prompt"] + cmd_args

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


@pytest.mark.smoke_test
def test_llm_neuralwatt_basic_connectivity():
    """Basic API connectivity smoke test."""
    try:
        result = _run_llm_command(
            [
                "Say hello world in one word",
                "-m",
                "neuralwatt-gpt-oss",
                "--no-stream",
                "--no-log",
            ]
        )

        # Check that the command executed successfully
        assert result.returncode == 0, (
            f"LLM command failed with return code {result.returncode}: {result.stderr}"
        )

        # Check that we got a response
        assert len(result.stdout.strip()) > 0, "No response received"

    except subprocess.TimeoutExpired:
        pytest.fail("LLM command timed out - NeuralWatt API may be unreachable")
    except Exception as e:
        pytest.fail(f"Unexpected error during basic connectivity test: {e}")


@pytest.mark.smoke_test
def test_llm_neuralwatt_streaming_functionality():
    """Streaming API functionality smoke test."""
    try:
        result = _run_llm_command(
            ["Say hello", "-m", "neuralwatt-gpt-oss", "--no-log"], timeout=30
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
        pytest.fail(f"Unexpected error during streaming test: {e}")


@pytest.mark.smoke_test
def test_llm_neuralwatt_streaming_vs_non_streaming():
    """Compare streaming vs non-streaming responses."""
    try:
        # Test with streaming (default)
        streaming_result = _run_llm_command(
            ["What is the capital of France?", "-m", "neuralwatt-gpt-oss", "--no-log"]
        )

        # Check that the streaming command executed successfully
        assert streaming_result.returncode == 0, (
            f"Streaming command failed with return code {streaming_result.returncode}: {streaming_result.stderr}"
        )

        # Test with explicit non-streaming
        non_streaming_result = _run_llm_command(
            [
                "What is the capital of France?",
                "-m",
                "neuralwatt-gpt-oss",
                "--no-stream",
                "--no-log",
            ]
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


@pytest.mark.smoke_test
def test_llm_neuralwatt_model_availability():
    """Verify that neuralwatt models are available."""
    try:
        # List available models
        cmd = [sys.executable, "-m", "llm", "models", "list"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        # Check that the command executed successfully
        assert result.returncode == 0, (
            f"LLM models command failed with return code {result.returncode}: {result.stderr}"
        )

        # Check that neuralwatt models are available
        output = result.stdout
        assert "neuralwatt/deepseek-coder-33b-instruct" in output, (
            "Expected model neuralwatt/deepseek-coder-33b-instruct not found"
        )
        assert "neuralwatt/gpt-oss-20b" in output, (
            "Expected model neuralwatt/gpt-oss-20b not found"
        )
        assert "neuralwatt/Qwen3-Coder-480B-A35B-Instruct" in output, (
            "Expected model neuralwatt/Qwen3-Coder-480B-A35B-Instruct not found"
        )

    except subprocess.TimeoutExpired:
        pytest.fail("LLM models command timed out")
    except Exception as e:
        pytest.fail(f"Unexpected error during models verification: {e}")
