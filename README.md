# Claude Code Azure GPT Proxy ([中文](./README.zh-CN.md))

> **Summary**
> This project proxies Anthropic Claude Code Messages API requests to Azure OpenAI `chat/completions` (and Responses where applicable), and converts responses back to Anthropic-compatible format. It supports SSE streaming and tool calls.

---

## Features

- **Protocol adaptation**: Convert Anthropic Messages API to Azure OpenAI Chat/Responses requests
- **Response conversion**: Map Azure OpenAI responses back to Anthropic Messages format
- **SSE streaming**: `message_start / content_block_delta / message_stop` events
- **Tool calls**: `tool_use / tool_result` support
- **Token counting**: `/v1/messages/count_tokens` local estimation

---

## Run locally

### 1. Prepare environment variables

Copy `.env.sample` to `.env` and fill in values:

```bash
copy .env.sample .env
```

### 2. Start the service

```bash
# Windows (PowerShell)
./start.ps1
```

The listening address is determined by `ASPNETCORE_URLS`. The startup log prints the final URL(s).

> Note: `start.ps1` loads `.env` and sets process-level environment variables.

---

## Docker build and run

### 1. Build image

```bash
docker build -t claude-azure-gpt-proxy .
```

### 2. Prepare environment variables

Copy `.env.sample` to `.env` and fill in values:

```bash
copy .env.sample .env
```

### 3. Run container

```bash
docker run -d --name claude-azure-gpt-proxy --env-file .env -p 8088:8080 claude-azure-gpt-proxy:latest
```

---

## Environment variables

| Name | Description |
|------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint (required) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key (required) |
| `AZURE_API_VERSION` | API version (e.g. `2024-10-21`) |
| `ANTHROPIC_AUTH_TOKEN` | If set, `/v1/messages*` requires Bearer token |
| `SMALL_MODEL` | Small model deployment name (default for haiku) |
| `BIG_MODEL` | Large model deployment name (default for sonnet/opus) |

---

## API

### `POST /v1/messages`

- Anthropic Messages API compatible
- Supports `stream=true` SSE

### `POST /v1/messages/count_tokens`

- Local token estimation
- Does not trigger generation

---

## License

MIT
