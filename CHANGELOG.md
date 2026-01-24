# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-01-24
### Added
- Streaming support: Energy data is now captured in both streaming and non-streaming modes, using a custom SSE decoder to extract energy usage from Neuralwatt's SSE comments.
- Async and sync model support for all Neuralwatt models.
- Documentation updates describing streaming and non-streaming energy data capture.

### Fixed
- Correctly logs energy data in `response_json` for both streaming and non-streaming completions.
- Improved compatibility with LLM's logging and model registration system.

### Changed
- Refactored code to share logic between sync and async model classes.
- Enhanced error handling and robustness for API key management and response parsing.

## [0.0.1] - 2025-01-21
### Added
- Initial release: Plugin for [LLM](https://llm.datasette.io/) to add support for Neuralwatt's OpenAI-compatible API.
- Captures and logs energy consumption data from Neuralwatt API responses in non-streaming mode.
- Registers Neuralwatt models with LLM, including aliases and API key management.
- Basic documentation and usage instructions.

[0.0.2]: https://github.com/mrchrisadams/llm-neuralwatt/releases/tag/0.0.2
[0.0.1]: https://github.com/mrchrisadams/llm-neuralwatt/releases/tag/0.0.1
