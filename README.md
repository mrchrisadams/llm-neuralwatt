# llm-neuralwatt

[![PyPI](https://img.shields.io/pypi/v/llm-neuralwatt.svg)](https://pypi.org/project/llm-neuralwatt/)
[![Changelog](https://img.shields.io/github/v/release/mrchrisadams/llm-neuralwatt?include_prereleases&label=changelog)](https://github.com/mrchrisadams/llm-neuralwatt/releases)
[![Tests](https://github.com/mrchrisadams/llm-neuralwatt/actions/workflows/test.yml/badge.svg)](https://github.com/mrchrisadams/llm-neuralwatt/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mrchrisadams/llm-neuralwatt/blob/main/LICENSE)

An plugin to add support for the OpenAI compatible Neuralwatt inference service, to run inference against various open weights models, and return the direct energy measurements in the response as well as logging it locally in llm's logs.db sqlite file.

This is not an official plugin from Neuralwatt.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-neuralwatt
```

## Usage

Once you have llm-neuralwatt installed, you should see new models available when you call `llm models`:

```
Neuralwatt: neuralwatt/deepseek-coder-33b-instruct (aliases: neuralwatt-deepseek-coder)
Neuralwatt: neuralwatt/gpt-oss-20b (aliases: neuralwatt-gpt-oss)
Neuralwatt: neuralwatt/Qwen3-Coder-480B-A35B-Instruct (aliases: neuralwatt-qwen3-coder)
```

You will need to set a key with:

```
llm keys set neuralwatt
```

You can sign up for Neuralwatt, and get an API key from https://portal.neuralwatt.com.

### Energy Consumption Logging

This plugin automatically captures and logs energy consumption data from Neuralwatt API responses. Energy data is stored in the `response_json` field of the llm logs database.

To view energy consumption for your requests:
```bash
# View recent logs with energy data
llm logs --model neuralwatt-gpt-oss --json | jq '.[-5:].response_json.energy'

# Query specific energy metrics
llm logs --model neuralwatt-deepseek-coder --json | jq -r '.[] | select(.response_json.energy != null) | "\(.datetime_utc): \(.response_json.energy.energy_joules) joules, \(.response_json.energy.energy_kwh) kWh"'
```

Each energy measurement includes:
- `energy_joules`: Energy consumption in joules
- `energy_kwh`: Energy consumption in kilowatt-hours  
- `avg_power_watts`: Average power consumption in watts
- `duration_seconds`: Duration of the API call
- `attribution_method`: How energy was attributed
- `attribution_ratio`: Ratio of energy attribution

You can read more about how energy consumption is attributed to a single use in the [Neuralwatt docs](https://portal.neuralwatt.com/docs/energy-methodology)

### Streaming Support

Energy consumption data is captured in both streaming and non-streaming modes. The plugin uses a custom HTTP streaming implementation to capture the energy data that Neuralwatt sends as an SSE comment just before the `[DONE]` marker.

```bash
# Both streaming and non-streaming capture energy data
llm "Explain quantum computing" -m neuralwatt-gpt-oss
llm "Explain quantum computing" -m neuralwatt-gpt-oss --no-stream
```

For more details on how Neuralwatt handles streaming, see the [Neuralwatt streaming docs](https://portal.neuralwatt.com/docs/guides/streaming).

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-neuralwatt
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
