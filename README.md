# Claude Code Azure GPT Proxy

> **简要说明**
> 该项目用于将 Anthropic Claude Code 的 Messages API 请求代理到 Azure OpenAI `chat/completions` 端点，并在响应侧转换回 Anthropic 兼容格式（支持 SSE 流式响应与工具调用）。

---

## 🚀 功能简介

- **协议适配**：将 Anthropic Messages API 请求转换为 Azure OpenAI Chat/Responses 请求格式
- **响应转换**：将 Azure OpenAI 响应重新映射为 Anthropic Messages 格式
- **SSE 流式支持**：支持 `message_start / content_block_delta / message_stop` 事件流
- **Tool 调用支持**：支持 tool_use / tool_result
- **Token 统计支持**：支持 `/v1/messages/count_tokens` 本地估算

---

## 🏃‍♂️ 本地运行

### 1. 准备环境变量

复制 `.env.sample` 为 `.env` 并按需填写：

```bash
copy .env.sample .env
```

### 2. 运行服务

```bash
# Windows (PowerShell)
./start.ps1
```

默认监听地址取决于 `ASPNETCORE_URLS`，启动日志会输出监听地址。

> 说明：`start.ps1` 会读取 `.env` 并设置进程级环境变量。

---

## 📦 Docker 构建与运行

### 1. 构建镜像

```bash
docker build -t claude-azure-gpt-proxy .
```

### 2. 准备环境变量

复制 `.env.sample` 为 `.env` 并按需填写：

```bash
copy .env.sample .env
```

### 3. 运行容器

```bash
docker run --rm -p 8080:8080 --env-file .env \
  claude-azure-gpt-proxy
```

---

## ⚙️ 环境变量

| 变量名 | 说明 |
|--------|------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI 资源端点（必填） |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI Key（必填） |
| `AZURE_API_VERSION` | API 版本（如 `2024-10-21`）|
| `ANTHROPIC_AUTH_TOKEN` | 若设置，则 `/v1/messages*` 需要 Bearer Token |
| `SMALL_MODEL` | 小模型部署名（默认用于 haiku）|
| `BIG_MODEL` | 大模型部署名（默认用于 sonnet/opus）|

---

## 🔌 接口说明

### `POST /v1/messages`

- Anthropic Messages API 兼容
- 支持 `stream=true` SSE

### `POST /v1/messages/count_tokens`

- 本地估算 token 数量
- 不触发真实生成

---

## 🔒 License

MIT
