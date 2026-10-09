# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `MODEL_PROVIDER` environment variable in `lib/agents.py` → `get_model()`, so the agents can run
  against a provider other than OpenAI. `openai` (default) keeps the existing behaviour and also
  covers a local OpenAI-compatible server (Ollama, vLLM, LM Studio, text-generation-webui) via
  `MODEL_API_BASE`. `huggingface` uses `InferenceClientModel`, for Hugging Face's Inference
  Providers or a local text-generation-inference endpoint running a model such as
  `meta-llama/Llama-2-70b-chat-hf`. `MODEL_ID` overrides the model name for either provider.
- `README.md`: a "Choosing a model provider" section documenting the new environment variables
  and a local-Ollama example.
