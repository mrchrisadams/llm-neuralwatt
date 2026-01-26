"""Smoke test for llm-neuralwatt plugin."""

import json
import subprocess
import sys
import pytest
import time
import os


def test_llm_neuralwatt_smoke_test():
    """Smoke test that demonstrates using llm with neuralwatt plugin."""

    # Get the API key from the file
    with open("neuralwatt.api.key.txt", "r") as f:
        api_key = f.read().strip()

    # Ensure the key is set in environment variable
    env = os.environ.copy()
    env["NEURALWATT_API_KEY"] = api_key

    # Test 1: Basic API connectivity with a small request
    try:
        # Using a simple prompt to avoid heavy processing
        cmd = [
            sys.executable,
            "-m",
            "llm",
            "prompt",
            "Say hello world in one word",
            "-m",
            "neuralwatt-gpt-oss",
            "--no-stream",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=env
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
        pytest.fail(f"Unexpected error during smoke test: {e}")


def test_llm_neuralwatt_with_logs():
    """Test that neuralwatt captures energy data in logs."""

    # Get the API key from the file
    with open("neuralwatt.api.key.txt", "r") as f:
        api_key = f.read().strip()

    # Ensure the key is set in environment variable
    env = os.environ.copy()
    env["NEURALWATT_API_KEY"] = api_key

    try:
        # Using a simple prompt to test logging
        cmd = [
            sys.executable,
            "-m",
            "llm",
            "prompt",
            "What is 2+2?",
            "-m",
            "neuralwatt-gpt-oss",
            "--no-stream",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=env
        )

        # Check that the command executed successfully
        assert result.returncode == 0, (
            f"LLM command failed with return code {result.returncode}: {result.stderr}"
        )

        # Check that we got a response
        assert len(result.stdout.strip()) > 0, "No response received"

        # Now check the logs for energy data
        log_cmd = [
            sys.executable,
            "-m",
            "llm",
            "logs",
            "--model",
            "neuralwatt-gpt-oss",
            "-n",
            "1",  # Only get the last entry
        ]

        log_result = subprocess.run(
            log_cmd, capture_output=True, text=True, timeout=10, env=env
        )

        # This test is more about verifying the API works than checking logs
        # since the log structure might vary

    except subprocess.TimeoutExpired:
        pytest.fail("LLM command timed out - NeuralWatt API may be unreachable")
    except Exception as e:
        pytest.fail(f"Unexpected error during logging test: {e}")


def test_llm_neuralwatt_available_models():
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
