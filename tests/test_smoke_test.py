"""Smoke tests for llm-neuralwatt plugin."""

import subprocess
import sys
import pytest
import os
import tempfile
import json


def _get_api_key():
    """Get the API key from the file."""
    with open("neuralwatt.api.key.txt", "r") as f:
        return f.read().strip()


def _run_llm_command(cmd_args, timeout=30, use_api_key=True, database_file=None):
    """Helper to run LLM commands with proper environment setup."""
    env = os.environ.copy()
    if use_api_key:
        env["NEURALWATT_API_KEY"] = _get_api_key()

    cmd = [sys.executable, "-m", "llm", "prompt"] + cmd_args

    # Add database file option if specified
    if database_file:
        cmd.extend(["-d", database_file])

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


@pytest.mark.smoke_test
def test_llm_neuralwatt_basic_connectivity():
    """Basic API connectivity smoke test."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db_file = tmp_db.name

        try:
            result = _run_llm_command(
                [
                    "Say hello world in one word",
                    "-m",
                    "neuralwatt-gpt-oss",
                    "--no-stream",
                ],
                database_file=db_file,
            )

            # Check that the command executed successfully
            assert result.returncode == 0, (
                f"LLM command failed with return code {result.returncode}: {result.stderr}"
            )

            # Check that we got a response
            assert len(result.stdout.strip()) > 0, "No response received"

        finally:
            # Clean up the temporary database file
            if os.path.exists(db_file):
                os.unlink(db_file)

    except subprocess.TimeoutExpired:
        pytest.fail("LLM command timed out - NeuralWatt API may be unreachable")
    except Exception as e:
        pytest.fail(f"Unexpected error during basic connectivity test: {e}")


@pytest.mark.smoke_test
def test_llm_neuralwatt_streaming_functionality():
    """Streaming API functionality smoke test."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db_file = tmp_db.name

        try:
            result = _run_llm_command(
                ["Say hello", "-m", "neuralwatt-gpt-oss"],
                timeout=30,
                database_file=db_file,
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

        finally:
            # Clean up the temporary database file
            if os.path.exists(db_file):
                os.unlink(db_file)

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
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db_file = tmp_db.name

        try:
            # Test with streaming (default)
            streaming_result = _run_llm_command(
                ["What is the capital of France?", "-m", "neuralwatt-gpt-oss"],
                database_file=db_file,
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
                ],
                database_file=db_file,
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
            assert (
                "Paris" in streaming_response or "paris" in streaming_response.lower()
            ), "Streaming response doesn't mention Paris"
            assert (
                "Paris" in non_streaming_response
                or "paris" in non_streaming_response.lower()
            ), "Non-streaming response doesn't mention Paris"

        finally:
            # Clean up the temporary database file
            if os.path.exists(db_file):
                os.unlink(db_file)

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


@pytest.mark.smoke_test
def test_llm_neuralwatt_energy_logging():
    """Test that energy data is properly logged and can be queried."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db_file = tmp_db.name

        try:
            # Run a command to generate energy logs
            result = _run_llm_command(
                ["What is 2+2?", "-m", "neuralwatt-gpt-oss", "--no-stream"],
                database_file=db_file,
            )

            # Check that the command executed successfully
            assert result.returncode == 0, (
                f"LLM command failed with return code {result.returncode}: {result.stderr}"
            )

            # Query the logs using the commands from the README
            log_cmd = [
                sys.executable,
                "-m",
                "llm",
                "logs",
                "--model",
                "neuralwatt-gpt-oss",
                "--json",
                "-d",
                db_file,
            ]

            log_result = subprocess.run(
                log_cmd, capture_output=True, text=True, timeout=10
            )

            assert log_result.returncode == 0, (
                f"LLM logs command failed with return code {log_result.returncode}: {log_result.stderr}"
            )

            # Parse the log output
            logs_data = json.loads(log_result.stdout)

            # Verify we have logs with energy data
            assert len(logs_data) > 0, "No logs found"

            # Check that energy data is present in the logs
            energy_present = False
            for log_entry in logs_data:
                if (
                    "response_json" in log_entry
                    and "energy" in log_entry["response_json"]
                ):
                    energy_data = log_entry["response_json"]["energy"]
                    assert "energy_joules" in energy_data, (
                        "Missing energy_joules in energy data"
                    )
                    assert "energy_kwh" in energy_data, (
                        "Missing energy_kwh in energy data"
                    )
                    assert "avg_power_watts" in energy_data, (
                        "Missing avg_power_watts in energy data"
                    )
                    assert energy_data["energy_joules"] >= 0, (
                        "Energy joules should be non-negative"
                    )
                    energy_present = True
                    break

            assert energy_present, "No energy data found in logs"

            # Test the jq commands from the README
            # First jq command: View recent logs with energy data
            jq_cmd1 = ["jq", ".[-1:].[].response_json.energy"]
            jq_process1 = subprocess.Popen(
                jq_cmd1,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            jq_output1, jq_error1 = jq_process1.communicate(input=log_result.stdout)

            assert jq_process1.returncode == 0, f"First jq command failed: {jq_error1}"
            assert "energy_joules" in jq_output1, (
                "First jq command didn't return expected energy data"
            )

            # Second jq command: Query specific energy metrics
            jq_cmd2 = [
                "jq",
                "-r",
                ".[] | select(.response_json.energy != null) | "
                + '"\\(.datetime_utc): \\(.response_json.energy.energy_joules) joules, '
                + '\\(.response_json.energy.energy_kwh) kWh"',
            ]

            jq_process2 = subprocess.Popen(
                jq_cmd2,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            jq_output2, jq_error2 = jq_process2.communicate(input=log_result.stdout)

            assert jq_process2.returncode == 0, f"Second jq command failed: {jq_error2}"
            assert "joules" in jq_output2, (
                "Second jq command didn't return expected format"
            )

        finally:
            # Clean up the temporary database file
            if os.path.exists(db_file):
                os.unlink(db_file)

    except subprocess.TimeoutExpired:
        pytest.fail("Command timed out")
    except FileNotFoundError:
        pytest.skip("jq not found - skipping jq command tests")
    except Exception as e:
        pytest.fail(f"Unexpected error during energy logging test: {e}")
