using System.Text.Json;
using ClaudeAzureGptProxy.Infrastructure;
using ClaudeAzureGptProxy.Models;
using Microsoft.Extensions.Logging;

namespace ClaudeAzureGptProxy.Services;

public static class AnthropicConversion
{
    private static string? ExtractTextFromSystem(object? systemBlock)
    {
        if (systemBlock is null)
        {
            return null;
        }

        if (systemBlock is string systemText)
        {
            return systemText;
        }

        if (systemBlock is JsonElement element && element.ValueKind == JsonValueKind.Array)
        {
            return ExtractTextFromSystem(EnumerateJsonArray(element));
        }

        if (systemBlock is IEnumerable<object> list)
        {
            var buffer = new List<string>();
            foreach (var item in list)
            {
                if (item is JsonElement itemElement)
                {
                    if (itemElement.ValueKind == JsonValueKind.Object &&
                        itemElement.TryGetProperty("type", out var typeProp) &&
                        typeProp.GetString() == "text" &&
                        itemElement.TryGetProperty("text", out var textProp))
                    {
                        buffer.Add(textProp.GetString() ?? string.Empty);
                    }
                }
                else if (item is Dictionary<string, object?> dict &&
                         dict.TryGetValue("type", out var typeObj) &&
                         string.Equals(typeObj?.ToString(), "text", StringComparison.OrdinalIgnoreCase) &&
                         dict.TryGetValue("text", out var textObj))
                {
                    buffer.Add(textObj?.ToString() ?? string.Empty);
                }
            }

            var combined = string.Join("\n\n", buffer).Trim();
            return string.IsNullOrWhiteSpace(combined) ? null : combined;
        }

        return null;
    }

    private static string StripProviderPrefix(string model)
    {
        if (model.StartsWith("anthropic/", StringComparison.OrdinalIgnoreCase))
        {
            return model["anthropic/".Length..];
        }

        if (model.StartsWith("openai/", StringComparison.OrdinalIgnoreCase))
        {
            return model["openai/".Length..];
        }

        if (model.StartsWith("gemini/", StringComparison.OrdinalIgnoreCase))
        {
            return model["gemini/".Length..];
        }

        if (model.StartsWith("azure/", StringComparison.OrdinalIgnoreCase))
        {
            return model["azure/".Length..];
        }

        return model;
    }

    public static string ResolveAzureModel(MessagesRequest request, NormalizedAzureOpenAiOptions azureOptions)
    {
        return ResolveAzureModel(request.Model, azureOptions);
    }

    public static string ResolveAzureModel(TokenCountRequest request, NormalizedAzureOpenAiOptions azureOptions)
    {
        return ResolveAzureModel(request.Model, azureOptions);
    }

    public static string ResolveAzureModel(string model, NormalizedAzureOpenAiOptions azureOptions)
    {
        if (model.StartsWith("azure/", StringComparison.OrdinalIgnoreCase))
        {
            return model;
        }

        var normalizedModel = StripProviderPrefix(model).ToLowerInvariant();
        // 兼容 claude-3/3.5/4 等命名，按子串匹配 haiku/sonnet/opus 进行映射。
        string? deployment = null;
        if (normalizedModel.Contains("haiku", StringComparison.OrdinalIgnoreCase))
        {
            deployment = azureOptions.SmallModel;
        }
        else if (normalizedModel.Contains("sonnet", StringComparison.OrdinalIgnoreCase) ||
                 normalizedModel.Contains("opus", StringComparison.OrdinalIgnoreCase))
        {
            deployment = azureOptions.BigModel;
        }

        if (!string.IsNullOrWhiteSpace(deployment))
        {
            return $"azure/{deployment}";
        }

        return model;
    }

    private static object? GetField(object? obj, string name)
    {
        if (obj is null)
        {
            return null;
        }

        if (obj is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var prop))
            {
                return prop;
            }
        }

        if (obj is IDictionary<string, object?> dict && dict.TryGetValue(name, out var value))
        {
            return value;
        }

        return null;
    }

    private static IEnumerable<object> EnumerateJsonArray(JsonElement element)
    {
        foreach (var item in element.EnumerateArray())
        {
            yield return item;
        }
    }

    public static Dictionary<string, object?> ConvertAnthropicToAzure(
        MessagesRequest request,
        ILogger logger,
        NormalizedAzureOpenAiOptions azureOptions)
    {
        var messages = new List<Dictionary<string, object?>>();

        if (request.System is not null)
        {
            var systemText = ExtractTextFromSystem(request.System);
            if (!string.IsNullOrWhiteSpace(systemText))
            {
                messages.Add(new Dictionary<string, object?>
                {
                    ["role"] = "system",
                    ["content"] = systemText
                });
            }
        }

        foreach (var message in request.Messages)
        {
            if (message.Content is string textContent)
            {
                messages.Add(new Dictionary<string, object?>
                {
                    ["role"] = message.Role,
                    ["content"] = textContent
                });
                continue;
            }

            if (message.Content is JsonElement element && element.ValueKind == JsonValueKind.Array)
            {
                var contentBlocks = EnumerateJsonArray(element);
                if (message.Role == "user")
                {
                    AppendUserContent(messages, contentBlocks);
                }
                else
                {
                    AppendAssistantContent(messages, contentBlocks, logger);
                }
                continue;
            }

            if (message.Content is IEnumerable<object> list)
            {
                if (message.Role == "user")
                {
                    AppendUserContent(messages, list);
                }
                else
                {
                    AppendAssistantContent(messages, list, logger);
                }
                continue;
            }
        }

        var resolvedModel = request.ResolvedAzureModel ?? request.Model;
        var deploymentName = StripProviderPrefix(resolvedModel);
        var modelName = deploymentName.ToLowerInvariant();

        var azureRequest = new Dictionary<string, object?>
        {
            ["model"] = resolvedModel,
            ["messages"] = messages,
            ["stream"] = request.Stream
        };

        if (modelName.Contains("gpt-5", StringComparison.OrdinalIgnoreCase) ||
            modelName.Contains("o3", StringComparison.OrdinalIgnoreCase))
        {
            azureRequest["max_completion_tokens"] = request.MaxTokens;
            logger.LogDebug("Using max_completion_tokens={MaxTokens} for Azure model {Model}",
                request.MaxTokens, modelName);
            logger.LogDebug(
                "Skipping temperature parameter for {Model} (only supports default temperature=1)",
                modelName);
        }
        else
        {
            azureRequest["max_tokens"] = request.MaxTokens;
            azureRequest["temperature"] = request.Temperature;
            logger.LogDebug(
                "Using max_tokens={MaxTokens} and temperature={Temperature} for Azure model {Model}",
                request.MaxTokens, request.Temperature, modelName);
        }

        if (request.StopSequences is { Count: > 0 })
        {
            azureRequest["stop"] = request.StopSequences;
        }

        if (request.TopP.HasValue)
        {
            azureRequest["top_p"] = request.TopP.Value;
        }

        if (request.TopK.HasValue)
        {
            azureRequest["top_k"] = request.TopK.Value;
        }

        if (request.Tools is { Count: > 0 })
        {
            var openAiTools = new List<Dictionary<string, object?>>();
            foreach (var tool in request.Tools)
            {
                if (string.IsNullOrWhiteSpace(tool.Name))
                {
                    throw new InvalidOperationException("Tool name is required.");
                }

                openAiTools.Add(new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["function"] = new Dictionary<string, object?>
                    {
                        ["name"] = tool.Name,
                        ["description"] = tool.Description ?? string.Empty,
                        ["parameters"] = tool.InputSchema
                    }
                });
            }

            azureRequest["tools"] = openAiTools;
        }

        if (request.ToolChoice is not null)
        {
            azureRequest["tool_choice"] = request.ToolChoice.Type switch
            {
                "auto" => "auto",
                "any" => "any",
                "tool" when !string.IsNullOrWhiteSpace(request.ToolChoice.Name) =>
                    new Dictionary<string, object?>
                    {
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object?>
                        {
                            ["name"] = request.ToolChoice.Name
                        }
                    },
                _ => "auto"
            };
        }

        return azureRequest;
    }

    private static void AppendUserContent(List<Dictionary<string, object?>> messages, IEnumerable<object> content)
    {
        var pendingText = string.Empty;

        void FlushUserText()
        {
            if (!string.IsNullOrWhiteSpace(pendingText))
            {
                messages.Add(new Dictionary<string, object?>
                {
                    ["role"] = "user",
                    ["content"] = pendingText.Trim()
                });
            }

            pendingText = string.Empty;
        }

        foreach (var block in content)
        {
            string? blockType = null;
            object? blockObj = block;

            if (block is JsonElement element && element.ValueKind == JsonValueKind.Object)
            {
                if (element.TryGetProperty("type", out var typeProp))
                {
                    blockType = typeProp.GetString();
                }
                blockObj = element;
            }
            else if (block is IDictionary<string, object?> dict)
            {
                blockType = dict.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
            }

            switch (blockType)
            {
                case "text":
                {
                    var textValue = ExtractTextValue(blockObj);
                    pendingText += $"{textValue}\n";
                    break;
                }
                case "image":
                    pendingText += "[Image content - not displayed in text format]\n";
                    break;
                case "tool_result":
                {
                    var toolUseId = ExtractString(blockObj, "tool_use_id") ?? string.Empty;
                    var resultContent = GetField(blockObj, "content");
                    FlushUserText();
                    messages.Add(new Dictionary<string, object?>
                    {
                        ["role"] = "tool",
                        ["tool_call_id"] = toolUseId,
                        ["content"] = ParseToolResultContent(resultContent)
                    });
                    break;
                }
                case "tool_use":
                {
                    var toolName = ExtractString(blockObj, "name") ?? string.Empty;
                    pendingText += $"[Tool use: {toolName}]\n";
                    break;
                }
            }
        }

        FlushUserText();
    }

    private static void AppendAssistantContent(
        List<Dictionary<string, object?>> messages,
        IEnumerable<object> content,
        ILogger logger)
    {
        var assistantText = string.Empty;
        var toolCalls = new List<Dictionary<string, object?>>();

        foreach (var block in content)
        {
            string? blockType = null;
            object? blockObj = block;

            if (block is JsonElement element && element.ValueKind == JsonValueKind.Object)
            {
                if (element.TryGetProperty("type", out var typeProp))
                {
                    blockType = typeProp.GetString();
                }
                blockObj = element;
            }
            else if (block is IDictionary<string, object?> dict)
            {
                blockType = dict.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
            }

            switch (blockType)
            {
                case "text":
                {
                    var textValue = ExtractTextValue(blockObj);
                    assistantText += $"{textValue}\n";
                    break;
                }
                case "image":
                    assistantText += "[Image content - not displayed in text format]\n";
                    break;
                case "tool_use":
                {
                    var toolId = ExtractString(blockObj, "id") ?? $"tool_{Guid.NewGuid()}";
                    var name = ExtractString(blockObj, "name") ?? string.Empty;
                    var toolInput = GetField(blockObj, "input") ?? new Dictionary<string, object?>();
                    var arguments = NormalizeToolArguments(toolInput, logger);

                    toolCalls.Add(new Dictionary<string, object?>
                    {
                        ["id"] = toolId,
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object?>
                        {
                            ["name"] = name,
                            ["arguments"] = arguments
                        }
                    });
                    break;
                }
                case "tool_result":
                {
                    var resultContent = GetField(blockObj, "content");
                    assistantText += $"{ParseToolResultContent(resultContent)}\n";
                    break;
                }
            }
        }

        var assistantMessage = new Dictionary<string, object?>
        {
            ["role"] = "assistant",
            ["content"] = assistantText.Trim()
        };

        if (toolCalls.Count > 0)
        {
            assistantMessage["tool_calls"] = toolCalls;
        }

        messages.Add(assistantMessage);
    }

    public static MessagesResponse ConvertAzureToAnthropic(
        object? azureResponse,
        MessagesRequest originalRequest,
        ILogger logger)
    {
        if (IsResponsesPayload(azureResponse))
        {
            return ConvertResponsesToAnthropic(azureResponse, originalRequest, logger);
        }

        var responseId = ExtractString(azureResponse, "id") ?? $"msg_{Guid.NewGuid():N}";
        var usage = GetField(azureResponse, "usage");
        var choices = GetField(azureResponse, "choices");
        object? messageObj = null;
        string? finishReason = null;
        string contentText = string.Empty;
        object? toolCalls = null;

        if (choices is JsonElement choicesElement && choicesElement.ValueKind == JsonValueKind.Array &&
            choicesElement.GetArrayLength() > 0)
        {
            var firstChoice = choicesElement[0];
            if (firstChoice.TryGetProperty("message", out var messageElement) &&
                messageElement.ValueKind == JsonValueKind.Object)
            {
                messageObj = messageElement;
                contentText = messageElement.TryGetProperty("content", out var contentProp)
                    ? contentProp.GetString() ?? string.Empty
                    : string.Empty;
                toolCalls = messageElement.TryGetProperty("tool_calls", out var toolCallsProp) ? toolCallsProp : null;
            }

            finishReason = firstChoice.TryGetProperty("finish_reason", out var finishProp)
                ? finishProp.GetString()
                : null;
        }
        else if (choices is IEnumerable<object> choicesList)
        {
            var firstChoice = choicesList.FirstOrDefault();
            if (firstChoice is not null)
            {
                messageObj = GetField(firstChoice, "message");
                contentText = ExtractString(messageObj, "content") ?? string.Empty;
                toolCalls = GetField(messageObj, "tool_calls");
                finishReason = ExtractString(firstChoice, "finish_reason");
            }
        }

        var content = new List<Dictionary<string, object?>>();
        if (!string.IsNullOrWhiteSpace(contentText))
        {
            content.Add(new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["text"] = contentText
            });
        }

        if (toolCalls is not null)
        {
            IEnumerable<object> toolList = toolCalls switch
            {
                JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
                IEnumerable<object> list => list,
                _ => new[] { toolCalls }
            };

            foreach (var toolCall in toolList)
            {
                var function = GetField(toolCall, "function");
                var toolId = ExtractString(toolCall, "id") ?? $"tool_{Guid.NewGuid()}";
                var name = ExtractString(function, "name") ?? string.Empty;
                var arguments = ExtractToolArguments(function, logger);
                var input = arguments is string rawArguments ? ParseJsonOrRaw(rawArguments, logger) : arguments;

                content.Add(new Dictionary<string, object?>
                {
                    ["type"] = "tool_use",
                    ["id"] = toolId,
                    ["name"] = name,
                    ["input"] = input
                });
            }
        }

        if (content.Count == 0)
        {
            content.Add(new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["text"] = string.Empty
            });
        }

        var usageInfo = ExtractUsage(usage);
        var stopReason = MapStopReason(finishReason);
        var responseModel = originalRequest.OriginalModel ?? originalRequest.Model;
        responseModel = StripProviderPrefix(responseModel);

        return new MessagesResponse
        {
            Id = responseId,
            Model = responseModel,
            Role = "assistant",
            Content = content,
            StopReason = stopReason,
            StopSequence = null,
            Usage = usageInfo
        };
    }

    private static bool IsResponsesPayload(object? azureResponse)
    {
        return GetField(azureResponse, "output") is not null;
    }

    private static MessagesResponse ConvertResponsesToAnthropic(
        object? azureResponse,
        MessagesRequest originalRequest,
        ILogger logger)
    {
        var responseId = ExtractString(azureResponse, "id") ?? $"msg_{Guid.NewGuid():N}";
        var usage = GetField(azureResponse, "usage");
        var output = GetField(azureResponse, "output");
        var responseModel = originalRequest.OriginalModel ?? originalRequest.Model;
        responseModel = StripProviderPrefix(responseModel);

        var content = new List<Dictionary<string, object?>>();
        ExtractResponsesContent(output, logger, content);

        if (content.Count == 0)
        {
            content.Add(new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["text"] = string.Empty
            });
        }

        var usageInfo = ExtractUsage(usage);
        var stopReason = MapStopReason(ExtractString(azureResponse, "finish_reason") ??
                                       ExtractString(azureResponse, "stop_reason") ??
                                       ExtractString(azureResponse, "status"));

        return new MessagesResponse
        {
            Id = responseId,
            Model = responseModel,
            Role = "assistant",
            Content = content,
            StopReason = stopReason,
            StopSequence = null,
            Usage = usageInfo
        };
    }

    private static void ExtractResponsesContent(
        object? output,
        ILogger logger,
        List<Dictionary<string, object?>> content)
    {
        if (output is null)
        {
            return;
        }

        IEnumerable<object> outputItems = output switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
            IEnumerable<object> list => list,
            _ => Array.Empty<object>()
        };

        var textBuffer = new List<string>();
        var toolBlocks = new List<Dictionary<string, object?>>();

        foreach (var item in outputItems)
        {
            var itemType = ExtractString(item, "type");

            if (string.Equals(itemType, "message", StringComparison.OrdinalIgnoreCase))
            {
                var messageContent = GetField(item, "content");
                ExtractResponsesMessageParts(messageContent, logger, textBuffer, toolBlocks);
                continue;
            }

            if (string.Equals(itemType, "output_text", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(itemType, "text", StringComparison.OrdinalIgnoreCase))
            {
                var text = ExtractString(item, "text") ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(text))
                {
                    textBuffer.Add(text);
                }
                continue;
            }

            if (string.Equals(itemType, "tool_call", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(itemType, "function_call", StringComparison.OrdinalIgnoreCase))
            {
                var toolBlock = BuildToolUseBlock(item, logger);
                if (toolBlock is not null)
                {
                    toolBlocks.Add(toolBlock);
                }
            }
        }

        var combinedText = string.Join(string.Empty, textBuffer).Trim();
        if (!string.IsNullOrWhiteSpace(combinedText))
        {
            content.Add(new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["text"] = combinedText
            });
        }

        content.AddRange(toolBlocks);
    }

    private static void ExtractResponsesMessageParts(
        object? messageContent,
        ILogger logger,
        List<string> textBuffer,
        List<Dictionary<string, object?>> toolBlocks)
    {
        if (messageContent is null)
        {
            return;
        }

        IEnumerable<object> parts = messageContent switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
            IEnumerable<object> list => list,
            _ => Array.Empty<object>()
        };

        foreach (var part in parts)
        {
            var partType = ExtractString(part, "type");
            if (string.Equals(partType, "output_text", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(partType, "text", StringComparison.OrdinalIgnoreCase))
            {
                var text = ExtractString(part, "text") ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(text))
                {
                    textBuffer.Add(text);
                }
                continue;
            }

            if (string.Equals(partType, "tool_call", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(partType, "function_call", StringComparison.OrdinalIgnoreCase))
            {
                var toolBlock = BuildToolUseBlock(part, logger);
                if (toolBlock is not null)
                {
                    toolBlocks.Add(toolBlock);
                }
            }
        }
    }

    private static Dictionary<string, object?>? BuildToolUseBlock(object? toolCall, ILogger logger)
    {
        var function = GetField(toolCall, "function");
        var toolId = ExtractString(toolCall, "id") ?? $"tool_{Guid.NewGuid()}";
        var name = ExtractString(toolCall, "name") ?? ExtractString(function, "name") ?? string.Empty;
        var arguments = ExtractString(toolCall, "arguments") ?? ExtractString(function, "arguments");

        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        var input = arguments is string rawArguments ? ParseJsonOrRaw(rawArguments, logger) : null;

        return new Dictionary<string, object?>
        {
            ["type"] = "tool_use",
            ["id"] = toolId,
            ["name"] = name,
            ["input"] = input ?? new Dictionary<string, object?>()
        };
    }

    private static Usage ExtractUsage(object? usage)
    {
        if (usage is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            var inputTokens = element.TryGetProperty("prompt_tokens", out var prompt)
                ? prompt.GetInt32()
                : element.TryGetProperty("input_tokens", out var input)
                    ? input.GetInt32()
                    : 0;

            var outputTokens = element.TryGetProperty("completion_tokens", out var completion)
                ? completion.GetInt32()
                : element.TryGetProperty("output_tokens", out var output)
                    ? output.GetInt32()
                    : 0;

            return new Usage
            {
                InputTokens = inputTokens,
                OutputTokens = outputTokens,
                CacheCreationInputTokens = 0,
                CacheReadInputTokens = 0
            };
        }

        if (usage is IDictionary<string, object?> dict)
        {
            var inputTokens = ExtractInt(dict, "prompt_tokens");
            if (inputTokens == 0)
            {
                inputTokens = ExtractInt(dict, "input_tokens");
            }

            var outputTokens = ExtractInt(dict, "completion_tokens");
            if (outputTokens == 0)
            {
                outputTokens = ExtractInt(dict, "output_tokens");
            }

            return new Usage
            {
                InputTokens = inputTokens,
                OutputTokens = outputTokens,
                CacheCreationInputTokens = 0,
                CacheReadInputTokens = 0
            };
        }

        return new Usage
        {
            InputTokens = 0,
            OutputTokens = 0,
            CacheCreationInputTokens = 0,
            CacheReadInputTokens = 0
        };
    }

    private static string MapStopReason(string? finishReason)
    {
        return finishReason switch
        {
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            _ => "end_turn"
        };
    }

    private static string ExtractTextValue(object? blockObj)
    {
        if (blockObj is JsonElement element && element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty("text", out var textProp))
        {
            return textProp.GetString() ?? string.Empty;
        }

        if (blockObj is IDictionary<string, object?> dict && dict.TryGetValue("text", out var textValue))
        {
            return textValue?.ToString() ?? string.Empty;
        }

        return string.Empty;
    }

    private static string? ExtractString(object? obj, string property)
    {
        if (obj is JsonElement element && element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(property, out var prop))
        {
            return prop.GetString();
        }

        if (obj is IDictionary<string, object?> dict && dict.TryGetValue(property, out var value))
        {
            return value?.ToString();
        }

        return null;
    }

    private static int ExtractInt(IDictionary<string, object?> dict, string property)
    {
        if (!dict.TryGetValue(property, out var value) || value is null)
        {
            return 0;
        }

        if (value is int intValue)
        {
            return intValue;
        }

        if (int.TryParse(value.ToString(), out var parsed))
        {
            return parsed;
        }

        return 0;
    }

    private static object? ExtractToolArguments(object? function, ILogger logger)
    {
        var argumentsRaw = GetField(function, "arguments");
        if (argumentsRaw is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String)
            {
                var jsonText = element.GetString() ?? "{}";
                return ParseJsonOrRaw(jsonText, logger);
            }

            if (element.ValueKind == JsonValueKind.Object || element.ValueKind == JsonValueKind.Array)
            {
                return JsonSerializer.Deserialize<object>(element.GetRawText());
            }
        }

        if (argumentsRaw is string rawText)
        {
            return ParseJsonOrRaw(rawText, logger);
        }

        return argumentsRaw ?? new Dictionary<string, object?>();
    }

    private static object ParseJsonOrRaw(string rawText, ILogger logger)
    {
        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, object?>>(rawText)
                   ?? new Dictionary<string, object?>();
        }
        catch (JsonException)
        {
            logger.LogWarning("Failed to parse tool arguments as JSON: {Arguments}", rawText);
            return new Dictionary<string, object?> { ["raw"] = rawText };
        }
    }

    private static object NormalizeToolArguments(object? toolInput, ILogger logger)
    {
        if (toolInput is null)
        {
            return new Dictionary<string, object?>();
        }

        if (toolInput is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String)
            {
                return element.GetString() ?? "{}";
            }

            if (element.ValueKind == JsonValueKind.Object || element.ValueKind == JsonValueKind.Array)
            {
                return element.GetRawText();
            }
        }

        if (toolInput is string toolText)
        {
            return toolText;
        }

        try
        {
            return JsonSerializer.Serialize(toolInput);
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Failed to serialize tool input");
            return toolInput.ToString() ?? string.Empty;
        }
    }

    private static string ParseToolResultContent(object? content)
    {
        if (content is null)
        {
            return "No content provided";
        }

        if (content is string contentString)
        {
            return contentString;
        }

        if (content is JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.String => element.GetString() ?? string.Empty,
                JsonValueKind.Object => element.GetRawText(),
                JsonValueKind.Array => element.GetRawText(),
                _ => element.ToString()
            };
        }

        if (content is IEnumerable<object> list)
        {
            var result = new List<string>();
            foreach (var item in list)
            {
                if (item is JsonElement itemElement && itemElement.ValueKind == JsonValueKind.Object)
                {
                    if (itemElement.TryGetProperty("type", out var typeProp) &&
                        typeProp.GetString() == "text" &&
                        itemElement.TryGetProperty("text", out var textProp))
                    {
                        result.Add(textProp.GetString() ?? string.Empty);
                    }
                    else
                    {
                        result.Add(itemElement.GetRawText());
                    }
                }
                else if (item is IDictionary<string, object?> dict &&
                         dict.TryGetValue("type", out var typeObj) &&
                         string.Equals(typeObj?.ToString(), "text", StringComparison.OrdinalIgnoreCase))
                {
                    result.Add(dict.TryGetValue("text", out var textObj) ? textObj?.ToString() ?? string.Empty : string.Empty);
                }
                else if (item is string itemText)
                {
                    result.Add(itemText);
                }
                else
                {
                    result.Add(item?.ToString() ?? string.Empty);
                }
            }

            return string.Join("\n", result).Trim();
        }

        if (content is IDictionary<string, object?> contentDict)
        {
            if (contentDict.TryGetValue("type", out var typeObj) &&
                string.Equals(typeObj?.ToString(), "text", StringComparison.OrdinalIgnoreCase) &&
                contentDict.TryGetValue("text", out var textObj))
            {
                return textObj?.ToString() ?? string.Empty;
            }

            return JsonSerializer.Serialize(contentDict);
        }

        return content.ToString() ?? string.Empty;
    }
}
