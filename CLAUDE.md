# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介
这是python的移植版本，原始的python版本位于 old目录下。用于代理Anthropic Claude Code API请求到Azure OpenAI服务，支持流式响应与工具调用。只实现了Azure OpenAI的chat/completions端点，转换请求与响应格式以兼容Anthropic的Messages API。因为其余暂时不需要，所以没有实现。

## 常用命令（Windows）

- 还原依赖：`dotnet restore`
- 构建：`dotnet build`
- 运行服务：`dotnet run --project ClaudeCodeAzureGptProxy.csproj`
- 运行全部测试：`dotnet test`
- 运行单个测试（按过滤）：`dotnet test --filter "FullyQualifiedName~ConversionTests"`

## 运行配置

- 运行前需设置环境变量：
  - `AZURE_OPENAI_ENDPOINT`：Azure OpenAI 端点（支持带/不带 api-version，会在配置中标准化）
  - `AZURE_OPENAI_API_KEY`：Azure OpenAI 访问密钥
  - `AZURE_API_VERSION`：Azure OpenAI API 版本
  - `ANTHROPIC_AUTH_TOKEN`：用于校验 `Authorization: Bearer` 的入站请求

## 架构概览

- **入口与路由**：`Program.cs` 定义最小 API，提供 `/v1/messages` 与 `/v1/messages/count_tokens` 路由，并处理 Bearer 令牌校验与流式响应。`/v1/messages` 支持 SSE 流式输出与普通 JSON 响应。
- **配置与依赖注入**：`Infrastructure/AzureOpenAiConfig.cs` 从环境变量读取并标准化 Azure OpenAI 端点（处理 api-version）。
- **请求转换与代理**：
  - `Services/AnthropicConversion.cs` 负责 Anthropic Messages <-> Azure OpenAI Chat/Responses 的双向转换（system、messages、tools、tool_choice、usage、stop_reason）。
  - `Services/AzureOpenAiClientFactory.cs` 创建 Azure OpenAI SDK `OpenAIClient`。
  - `Services/AzureOpenAiProxy.cs` 发送普通请求与流式请求；将 Azure 响应转换回 Anthropic 格式。
- **SSE 事件封装**：`Services/SseStreaming.cs` 将 Azure 流式增量转为 Anthropic SSE 事件序列（message/content_block/message_stop）。
- **数据模型**：`Models/Messages.cs` 定义 Anthropic Messages 请求/响应、工具与 content block 结构。
- **测试**：`ClaudeCodeAzureGptProxy.Tests/ConversionTests.cs` 覆盖关键转换与 SSE 事件。