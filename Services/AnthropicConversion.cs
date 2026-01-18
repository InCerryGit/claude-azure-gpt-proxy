using System.Text.Json;
using System.Linq;
using AzureGptProxy.Models;
using AzureGptProxy.Infrastructure;
using Microsoft.Extensions.Logging;

namespace AzureGptProxy.Services;

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
                        if (HasSystemMetadata(itemElement))
                        {
                            buffer.Add(itemElement.GetRawText());
                        }
                        else
                        {
                            buffer.Add(textProp.GetString() ?? string.Empty);
                        }
                    }
                    else if (itemElement.ValueKind == JsonValueKind.Object)
                    {
                        buffer.Add(itemElement.GetRawText());
                    }
                }
                else if (item is Dictionary<string, object?> dict &&
                         dict.TryGetValue("type", out var typeObj) &&
                         string.Equals(typeObj?.ToString(), "text", StringComparison.OrdinalIgnoreCase) &&
                         dict.TryGetValue("text", out var textObj))
                {
                    if (HasSystemMetadata(dict))
                    {
                        buffer.Add(JsonSerializer.Serialize(dict));
                    }
                    else
                    {
                        buffer.Add(textObj?.ToString() ?? string.Empty);
                    }
                }
                else if (item is Dictionary<string, object?> otherDict)
                {
                    buffer.Add(JsonSerializer.Serialize(otherDict));
                }
            }

            var combined = string.Join("\n\n", buffer).Trim();
            return string.IsNullOrWhiteSpace(combined) ? null : combined;
        }

        return null;
    }

    private static bool HasSystemMetadata(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        return element.TryGetProperty("cache_control", out _) ||
               element.TryGetProperty("citations", out _);
    }

    private static bool HasSystemMetadata(Dictionary<string, object?> dict)
    {
        return dict.ContainsKey("cache_control") || dict.ContainsKey("citations");
    }

    internal static string StripProviderPrefix(string model)
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

        foreach (var message in MergeMessages(request.Messages))
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
                var toolType = string.IsNullOrWhiteSpace(tool.Type) ? "function" : tool.Type;

                if (!string.Equals(toolType, "function", StringComparison.OrdinalIgnoreCase))
                {
                    var passthrough = new Dictionary<string, object?>
                    {
                        ["type"] = toolType
                    };

                    if (!string.IsNullOrWhiteSpace(tool.Name))
                    {
                        passthrough["name"] = tool.Name;
                    }

                    if (!string.IsNullOrWhiteSpace(tool.Description))
                    {
                        passthrough["description"] = tool.Description;
                    }

                    if (tool.InputSchema.Count > 0)
                    {
                        passthrough["parameters"] = tool.InputSchema;
                    }

                    if (tool.AdditionalProperties is not null)
                    {
                        foreach (var (key, value) in tool.AdditionalProperties)
                        {
                            if (!passthrough.ContainsKey(key))
                            {
                                passthrough[key] = value;
                            }
                        }
                    }

                    openAiTools.Add(passthrough);
                    continue;
                }

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
                "none" => "none",
                "tool" when !string.IsNullOrWhiteSpace(request.ToolChoice.Name) =>
                    new Dictionary<string, object?>
                    {
                        ["type"] = "tool",
                        ["name"] = request.ToolChoice.Name
                    },
                _ => "auto"
            };

            if (request.ToolChoice.DisableParallelToolUse == true)
            {
                // OpenAI-style flag; only effective on Responses path.
                azureRequest["parallel_tool_calls"] = false;
            }
        }

        if (request.Metadata is not null)
        {
            azureRequest["metadata"] = request.Metadata;
        }

        if (!string.IsNullOrWhiteSpace(request.User))
        {
            azureRequest["user"] = request.User;
        }

        if (!string.IsNullOrWhiteSpace(request.PreviousResponseId))
        {
            azureRequest["previous_response_id"] = request.PreviousResponseId;
        }

        if (request.Background.HasValue)
        {
            azureRequest["background"] = request.Background.Value;
        }

        if (request.Store.HasValue)
        {
            azureRequest["store"] = request.Store.Value;
        }
        else if (request.Background == true)
        {
            // Responses background mode requires store=true.
            azureRequest["store"] = true;
        }

        if (request.Include is { Count: > 0 })
        {
            azureRequest["include"] = request.Include;
        }

        if (!string.IsNullOrWhiteSpace(request.Truncation))
        {
            azureRequest["truncation"] = request.Truncation;
        }

        if (request.Thinking is not null && request.Thinking.IsEnabled())
        {
            if (modelName.Contains("gpt-5", StringComparison.OrdinalIgnoreCase) ||
                modelName.Contains("o3", StringComparison.OrdinalIgnoreCase))
            {
                azureRequest["reasoning"] = new Dictionary<string, object?>
                {
                    ["effort"] = "medium"
                };
            }
        }

        return azureRequest;
    }

    private static List<Message> MergeMessages(List<Message> messages)
    {
        var merged = new List<Message>();
        foreach (var message in messages)
        {
            if (merged.Count == 0)
            {
                merged.Add(message);
                continue;
            }

            var last = merged[^1];
            if (!string.Equals(last.Role, message.Role, StringComparison.OrdinalIgnoreCase))
            {
                merged.Add(message);
                continue;
            }

            var combinedContent = MergeMessageContent(last.Content, message.Content);
            merged[^1] = last with { Content = combinedContent };
        }

        return merged;
    }

    private static object? MergeMessageContent(object? left, object? right)
    {
        if (left is null)
        {
            return right;
        }

        if (right is null)
        {
            return left;
        }

        if (left is string leftText && right is string rightText)
        {
            return string.Concat(leftText, "\n\n", rightText);
        }

        var blocks = new List<object>();
        blocks.AddRange(ToContentBlocks(left));
        blocks.AddRange(ToContentBlocks(right));
        return blocks;
    }

    private static IEnumerable<object> ToContentBlocks(object content)
    {
        if (content is string text)
        {
            yield return new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["text"] = text
            };
            yield break;
        }

        if (content is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in element.EnumerateArray())
                {
                    yield return item;
                }
                yield break;
            }

            if (element.ValueKind == JsonValueKind.Object)
            {
                yield return element;
                yield break;
            }

            if (element.ValueKind == JsonValueKind.String)
            {
                yield return new Dictionary<string, object?>
                {
                    ["type"] = "text",
                    ["text"] = element.GetString() ?? string.Empty
                };
                yield break;
            }
        }

        if (content is IEnumerable<object> list)
        {
            foreach (var item in list)
            {
                yield return item;
            }
            yield break;
        }

        yield return new Dictionary<string, object?>
        {
            ["type"] = "text",
            ["text"] = content.ToString() ?? string.Empty
        };
    }

    private static void AppendUserContent(List<Dictionary<string, object?>> messages, IEnumerable<object> content)
    {
        var pendingText = string.Empty;
        var contentBlocks = new List<Dictionary<string, object?>>();
        var useBlocks = false;

        void FlushUserTextToBlocks()
        {
            if (string.IsNullOrWhiteSpace(pendingText))
            {
                return;
            }

            contentBlocks.Add(new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["text"] = pendingText.Trim()
            });

            pendingText = string.Empty;
        }

        void FlushUserBlocks()
        {
            if (contentBlocks.Count == 0)
            {
                return;
            }

            messages.Add(new Dictionary<string, object?>
            {
                ["role"] = "user",
                ["content"] = new List<Dictionary<string, object?>>(contentBlocks)
            });
            contentBlocks.Clear();
        }

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
                    if (useBlocks)
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = textValue
                        });
                    }
                    else
                    {
                        pendingText += $"{textValue}\n";
                    }
                    break;
                }
                case "image":
                    useBlocks = true;
                    FlushUserTextToBlocks();
                    if (TryConvertBlockToDictionary(blockObj, out var imageBlock))
                    {
                        contentBlocks.Add(imageBlock);
                    }
                    else
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = SerializeContentBlock(blockObj)
                        });
                    }
                    break;
                case "document":
                    useBlocks = true;
                    FlushUserTextToBlocks();
                    if (TryConvertBlockToDictionary(blockObj, out var documentBlock))
                    {
                        contentBlocks.Add(documentBlock);
                    }
                    else
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = SerializeContentBlock(blockObj)
                        });
                    }
                    break;
                case "input_file":
                case "file":
                    useBlocks = true;
                    FlushUserTextToBlocks();
                    if (TryConvertBlockToDictionary(blockObj, out var fileBlock))
                    {
                        contentBlocks.Add(fileBlock);
                    }
                    else
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = SerializeContentBlock(blockObj)
                        });
                    }
                    break;
                case "search_result":
                    useBlocks = true;
                    FlushUserTextToBlocks();
                    if (TryConvertBlockToDictionary(blockObj, out var searchBlock))
                    {
                        contentBlocks.Add(searchBlock);
                    }
                    else
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = SerializeContentBlock(blockObj)
                        });
                    }
                    break;
                case "tool_result":
                {
                    var toolUseId = ExtractString(blockObj, "tool_use_id") ?? string.Empty;
                    var resultContent = GetField(blockObj, "content");
                    var isError = ExtractBool(blockObj, "is_error");
                    if (useBlocks)
                    {
                        FlushUserTextToBlocks();
                        FlushUserBlocks();
                    }
                    else
                    {
                        FlushUserText();
                    }
                    messages.Add(new Dictionary<string, object?>
                    {
                        ["role"] = "tool",
                        ["tool_call_id"] = toolUseId,
                        ["content"] = BuildToolResultPayload(resultContent, isError)
                    });
                    break;
                }
                case "tool_use":
                {
                    if (useBlocks)
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = SerializeContentBlock(blockObj)
                        });
                    }
                    else
                    {
                        pendingText += $"{SerializeContentBlock(blockObj)}\n";
                    }
                    break;
                }
                default:
                    if (useBlocks)
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = SerializeContentBlock(blockObj)
                        });
                    }
                    else
                    {
                        pendingText += $"{SerializeContentBlock(blockObj)}\n";
                    }
                    break;
            }
        }

        if (useBlocks)
        {
            FlushUserTextToBlocks();
            FlushUserBlocks();
        }
        else
        {
            FlushUserText();
        }
    }

    private static void AppendAssistantContent(
        List<Dictionary<string, object?>> messages,
        IEnumerable<object> content,
        ILogger logger)
    {
        var assistantText = string.Empty;
        var contentBlocks = new List<Dictionary<string, object?>>();
        var useBlocks = false;
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
                    if (useBlocks)
                    {
                        contentBlocks.Add(new Dictionary<string, object?>
                        {
                            ["type"] = "text",
                            ["text"] = textValue
                        });
                    }
                    else
                    {
                        assistantText += $"{textValue}\n";
                    }
                    break;
                }
                case "image":
                    useBlocks = true;
                    contentBlocks.Add(new Dictionary<string, object?>
                    {
                        ["type"] = "text",
                        ["text"] = SerializeContentBlock(blockObj)
                    });
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
                    useBlocks = true;
                    contentBlocks.Add(new Dictionary<string, object?>
                    {
                        ["type"] = "text",
                        ["text"] = SerializeContentBlock(blockObj)
                    });
                    break;
                }
                case "tool_result":
                {
                    var resultContent = GetField(blockObj, "content");
                    useBlocks = true;
                    contentBlocks.Add(new Dictionary<string, object?>
                    {
                        ["type"] = "text",
                        ["text"] = ParseToolResultContent(resultContent)
                    });
                    break;
                }
                case "document":
                case "search_result":
                case "thinking":
                case "redacted_thinking":
                case "input_file":
                case "file":
                    useBlocks = true;
                    contentBlocks.Add(new Dictionary<string, object?>
                    {
                        ["type"] = "text",
                        ["text"] = SerializeContentBlock(blockObj)
                    });
                    break;
                default:
                    useBlocks = true;
                    contentBlocks.Add(new Dictionary<string, object?>
                    {
                        ["type"] = "text",
                        ["text"] = SerializeContentBlock(blockObj)
                    });
                    break;
            }
        }

        var assistantMessage = new Dictionary<string, object?>
        {
            ["role"] = "assistant",
            ["content"] = useBlocks ? contentBlocks : assistantText.Trim()
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
        foreach (var item in outputItems)
        {
            AppendResponsesOutputItem(item, logger, content);
        }
    }

    private static void ExtractResponsesMessageParts(
        object? messageContent,
        ILogger logger,
        List<Dictionary<string, object?>> content)
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
            AppendResponsesOutputItem(part, logger, content);
        }
    }

    private static void AppendResponsesOutputItem(
        object? item,
        ILogger logger,
        List<Dictionary<string, object?>> content)
    {
        if (item is null)
        {
            return;
        }

        var itemType = ExtractString(item, "type") ?? string.Empty;
        if (string.Equals(itemType, "message", StringComparison.OrdinalIgnoreCase))
        {
            var messageContent = GetField(item, "content");
            ExtractResponsesMessageParts(messageContent, logger, content);
            return;
        }

        if (string.Equals(itemType, "output_text", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(itemType, "text", StringComparison.OrdinalIgnoreCase))
        {
            var text = ExtractString(item, "text") ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(text))
            {
                content.Add(new Dictionary<string, object?>
                {
                    ["type"] = "text",
                    ["text"] = text
                });
            }
            return;
        }

        if (string.Equals(itemType, "reasoning", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(itemType, "reasoning_text", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(itemType, "thinking", StringComparison.OrdinalIgnoreCase))
        {
            var thinkingText = ExtractString(item, "thinking") ?? ExtractString(item, "text") ?? JsonSerializer.Serialize(item);
            content.Add(new Dictionary<string, object?>
            {
                ["type"] = "thinking",
                ["thinking"] = thinkingText,
                ["signature"] = ExtractString(item, "signature") ?? string.Empty
            });
            return;
        }

        if (string.Equals(itemType, "redacted_thinking", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(itemType, "redacted_reasoning", StringComparison.OrdinalIgnoreCase))
        {
            var data = ExtractString(item, "data") ?? ExtractString(item, "text") ?? JsonSerializer.Serialize(item);
            content.Add(new Dictionary<string, object?>
            {
                ["type"] = "redacted_thinking",
                ["data"] = data
            });
            return;
        }

        if (string.Equals(itemType, "tool_call", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(itemType, "function_call", StringComparison.OrdinalIgnoreCase))
        {
            var toolBlock = BuildToolUseBlock(item, logger);
            if (toolBlock is not null)
            {
                content.Add(toolBlock);
            }
            return;
        }

        // Fallback: preserve unknown Responses output items as text to avoid data loss.
        content.Add(new Dictionary<string, object?>
        {
            ["type"] = "text",
            ["text"] = JsonSerializer.Serialize(item)
        });
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
            // Azure/OpenAI usage shapes vary:
            // - Chat Completions: {prompt_tokens, completion_tokens, total_tokens}
            // - Responses: {input_tokens, output_tokens}
            // Some SDKs/proxies may omit one side; we try to recover from total_tokens when possible.

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

            // Azure Responses may include cached token counts under input_tokens_details.cached_tokens.
            // We map this to Anthropic's cache_read_input_tokens (best-effort).
            var cacheReadInputTokens = 0;
            if (element.TryGetProperty("input_tokens_details", out var inputDetails) &&
                inputDetails.ValueKind == JsonValueKind.Object &&
                inputDetails.TryGetProperty("cached_tokens", out var cachedTokensProp) &&
                cachedTokensProp.ValueKind == JsonValueKind.Number)
            {
                cacheReadInputTokens = cachedTokensProp.GetInt32();
            }

            if (inputTokens == 0 && outputTokens > 0 &&
                element.TryGetProperty("total_tokens", out var total) &&
                total.ValueKind == JsonValueKind.Number)
            {
                var totalTokens = total.GetInt32();
                // Best-effort: if only total is provided, attribute the remainder to input.
                inputTokens = Math.Max(0, totalTokens - outputTokens);
            }

            if (outputTokens == 0 && inputTokens > 0 &&
                element.TryGetProperty("total_tokens", out var total2) &&
                total2.ValueKind == JsonValueKind.Number)
            {
                var totalTokens = total2.GetInt32();
                outputTokens = Math.Max(0, totalTokens - inputTokens);
            }

            return new Usage
            {
                CacheCreation = new Dictionary<string, object?>
                {
                    ["ephemeral_5m_input_tokens"] = 0,
                    ["ephemeral_1h_input_tokens"] = 0
                },
                InputTokens = inputTokens,
                OutputTokens = outputTokens,
                CacheCreationInputTokens = 0,
                CacheReadInputTokens = cacheReadInputTokens
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

            var cacheReadInputTokens = 0;
            if (dict.TryGetValue("input_tokens_details", out var inputDetailsObj) && inputDetailsObj is not null)
            {
                if (inputDetailsObj is JsonElement detailsEl && detailsEl.ValueKind == JsonValueKind.Object &&
                    detailsEl.TryGetProperty("cached_tokens", out var cachedTokensProp) &&
                    cachedTokensProp.ValueKind == JsonValueKind.Number)
                {
                    cacheReadInputTokens = cachedTokensProp.GetInt32();
                }
                else if (inputDetailsObj is IDictionary<string, object?> detailsDict)
                {
                    cacheReadInputTokens = ExtractInt(detailsDict, "cached_tokens");
                }
            }

            if (inputTokens == 0 && outputTokens > 0)
            {
                var totalTokens = ExtractInt(dict, "total_tokens");
                if (totalTokens > 0)
                {
                    inputTokens = Math.Max(0, totalTokens - outputTokens);
                }
            }

            if (outputTokens == 0 && inputTokens > 0)
            {
                var totalTokens = ExtractInt(dict, "total_tokens");
                if (totalTokens > 0)
                {
                    outputTokens = Math.Max(0, totalTokens - inputTokens);
                }
            }

            return new Usage
            {
                CacheCreation = new Dictionary<string, object?>
                {
                    ["ephemeral_5m_input_tokens"] = 0,
                    ["ephemeral_1h_input_tokens"] = 0
                },
                InputTokens = inputTokens,
                OutputTokens = outputTokens,
                CacheCreationInputTokens = 0,
                CacheReadInputTokens = cacheReadInputTokens
            };
        }

        return new Usage
        {
            CacheCreation = new Dictionary<string, object?>
            {
                ["ephemeral_5m_input_tokens"] = 0,
                ["ephemeral_1h_input_tokens"] = 0
            },
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
            "stop" => "end_turn",
            "content_filter" => "content_filter",
            "cancelled" => "cancelled",
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

    private static bool? ExtractBool(object? obj, string property)
    {
        if (obj is JsonElement element && element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(property, out var prop))
        {
            if (prop.ValueKind == JsonValueKind.True)
            {
                return true;
            }

            if (prop.ValueKind == JsonValueKind.False)
            {
                return false;
            }

            if (prop.ValueKind == JsonValueKind.String && bool.TryParse(prop.GetString(), out var parsed))
            {
                return parsed;
            }
        }

        if (obj is IDictionary<string, object?> dict && dict.TryGetValue(property, out var value) && value is not null)
        {
            if (value is bool boolValue)
            {
                return boolValue;
            }

            if (bool.TryParse(value.ToString(), out var parsed))
            {
                return parsed;
            }
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
            var allTextBlocks = true;
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
                        allTextBlocks = false;
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
                    allTextBlocks = false;
                    result.Add(item?.ToString() ?? string.Empty);
                }
            }

            if (!allTextBlocks)
            {
                return JsonSerializer.Serialize(list);
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

    private static string BuildToolResultPayload(object? content, bool? isError)
    {
        var parsed = ParseToolResultContent(content);
        if (isError != true)
        {
            return parsed;
        }

        var payload = new Dictionary<string, object?>
        {
            ["is_error"] = true,
            ["content"] = parsed
        };

        return JsonSerializer.Serialize(payload);
    }

    private static string SerializeContentBlock(object? blockObj)
    {
        if (blockObj is null)
        {
            return string.Empty;
        }

        if (blockObj is JsonElement element)
        {
            return element.ValueKind == JsonValueKind.String
                ? element.GetString() ?? string.Empty
                : element.GetRawText();
        }

        if (blockObj is string str)
        {
            return str;
        }

        if (blockObj is IDictionary<string, object?> dict)
        {
            return JsonSerializer.Serialize(dict);
        }

        return JsonSerializer.Serialize(blockObj);
    }

    private static bool TryConvertBlockToDictionary(object? blockObj, out Dictionary<string, object?> block)
    {
        block = new Dictionary<string, object?>();

        if (blockObj is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Object)
            {
                block = ConvertJsonObjectToDictionary(element);
                return true;
            }

            return false;
        }

        if (blockObj is IDictionary<string, object?> dict)
        {
            block = new Dictionary<string, object?>(dict);
            return true;
        }

        return false;
    }

    private static Dictionary<string, object?> ConvertJsonObjectToDictionary(JsonElement element)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var prop in element.EnumerateObject())
        {
            dict[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.Object => ConvertJsonObjectToDictionary(prop.Value),
                JsonValueKind.Array => prop.Value.EnumerateArray().Select(item => (object)item).ToList(),
                JsonValueKind.String => prop.Value.GetString(),
                JsonValueKind.Number => prop.Value.TryGetInt64(out var l) ? l : prop.Value.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null
            };
        }

        return dict;
    }
}
