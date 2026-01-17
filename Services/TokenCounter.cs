using System.Text.Json;
using ClaudeAzureGptProxy.Models;
using Microsoft.Extensions.Logging;
using SharpToken;

namespace ClaudeAzureGptProxy.Services;

public sealed class TokenCounter
{
    private const string ImagePlaceholder = "[Image content - not displayed in text format]";
    private const string ThinkingPlaceholder = "[thinking enabled]";
    private const string FallbackEncodingName = "cl100k_base";
    private const string LargeEncodingName = "o200k_base";

    private readonly ILogger<TokenCounter> _logger;

    public TokenCounter(ILogger<TokenCounter> logger)
    {
        _logger = logger;
    }

    public int CountInputTokens(TokenCountRequest request)
    {
        var encodingName = SelectEncodingName(request.Model);
        var encoding = GetEncoding(encodingName, out var usedFallback);

        if (usedFallback)
        {
            _logger.LogWarning("Tokenizer encoding {RequestedEncoding} unavailable; using {FallbackEncoding}",
                encodingName, FallbackEncodingName);
        }

        var total = 0;
        total += CountSystemTokens(request.System, encoding);
        total += CountMessagesTokens(request.Messages, encoding);
        total += CountToolsTokens(request.Tools, encoding);
        total += CountToolChoiceTokens(request.ToolChoice, encoding);
        total += CountThinkingTokens(request.Thinking, encoding);

        _logger.LogInformation("Counted {TokenCount} input tokens for model {Model} using {Encoding}",
            total, request.Model, usedFallback ? FallbackEncodingName : encodingName);

        return total;
    }

    private static string SelectEncodingName(string model)
    {
        if (string.IsNullOrWhiteSpace(model))
        {
            return FallbackEncodingName;
        }

        var normalized = model.ToLowerInvariant();
        if (normalized.Contains("gpt-4o", StringComparison.OrdinalIgnoreCase) ||
            normalized.Contains("gpt-4.1", StringComparison.OrdinalIgnoreCase) ||
            normalized.Contains("o3", StringComparison.OrdinalIgnoreCase) ||
            normalized.Contains("gpt-5", StringComparison.OrdinalIgnoreCase))
        {
            return LargeEncodingName;
        }

        return FallbackEncodingName;
    }

    private static GptEncoding GetEncoding(string encodingName, out bool usedFallback)
    {
        try
        {
            usedFallback = false;
            return GptEncoding.GetEncoding(encodingName);
        }
        catch (Exception)
        {
            usedFallback = true;
            return GptEncoding.GetEncoding(FallbackEncodingName);
        }
    }

    private static int CountSystemTokens(object? systemBlock, GptEncoding encoding)
    {
        var systemText = ExtractTextFromSystem(systemBlock);
        if (string.IsNullOrWhiteSpace(systemText))
        {
            return 0;
        }

        return encoding.Encode(systemText).Count;
    }

    private static int CountMessagesTokens(IEnumerable<Message> messages, GptEncoding encoding)
    {
        var total = 0;
        foreach (var message in messages)
        {
            total += CountMessageTokens(message, encoding);
        }

        return total;
    }

    private static int CountMessageTokens(Message message, GptEncoding encoding)
    {
        if (message.Content is string textContent)
        {
            return encoding.Encode(textContent).Count;
        }

        if (message.Content is JsonElement element && element.ValueKind == JsonValueKind.Array)
        {
            return CountContentBlocks(EnumerateJsonArray(element), encoding);
        }

        if (message.Content is IEnumerable<object> list)
        {
            return CountContentBlocks(list, encoding);
        }

        return 0;
    }

    private static int CountContentBlocks(IEnumerable<object> contentBlocks, GptEncoding encoding)
    {
        var total = 0;
        foreach (var block in contentBlocks)
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
                    if (!string.IsNullOrWhiteSpace(textValue))
                    {
                        total += encoding.Encode(textValue).Count;
                    }
                    break;
                }
                case "image":
                    total += encoding.Encode(ImagePlaceholder).Count;
                    break;
                case "tool_use":
                {
                    var toolName = ExtractString(blockObj, "name") ?? string.Empty;
                    var toolInput = GetField(blockObj, "input");
                    var toolPayload = new Dictionary<string, object?>
                    {
                        ["name"] = toolName,
                        ["input"] = toolInput
                    };
                    var json = JsonSerializer.Serialize(toolPayload);
                    total += encoding.Encode(json).Count;
                    break;
                }
                case "tool_result":
                {
                    var toolUseId = ExtractString(blockObj, "tool_use_id") ?? string.Empty;
                    var resultContent = GetField(blockObj, "content");
                    var payload = new Dictionary<string, object?>
                    {
                        ["tool_use_id"] = toolUseId,
                        ["content"] = resultContent
                    };
                    var json = JsonSerializer.Serialize(payload);
                    total += encoding.Encode(json).Count;
                    break;
                }
            }
        }

        return total;
    }

    private static int CountToolsTokens(IEnumerable<Tool>? tools, GptEncoding encoding)
    {
        if (tools is null)
        {
            return 0;
        }

        var total = 0;
        foreach (var tool in tools)
        {
            var payload = new Dictionary<string, object?>
            {
                ["name"] = tool.Name,
                ["description"] = tool.Description ?? string.Empty,
                ["input_schema"] = tool.InputSchema
            };
            var json = JsonSerializer.Serialize(payload);
            total += encoding.Encode(json).Count;
        }

        return total;
    }

    private static int CountToolChoiceTokens(ToolChoice? toolChoice, GptEncoding encoding)
    {
        if (toolChoice is null)
        {
            return 0;
        }

        return toolChoice.Type switch
        {
            "auto" => encoding.Encode("auto").Count,
            "any" => encoding.Encode("any").Count,
            "tool" when !string.IsNullOrWhiteSpace(toolChoice.Name)
                => encoding.Encode(JsonSerializer.Serialize(new { type = "tool", name = toolChoice.Name })).Count,
            _ => 0
        };
    }

    private static int CountThinkingTokens(ThinkingConfig? thinking, GptEncoding encoding)
    {
        if (thinking?.Enabled != true)
        {
            return 0;
        }

        return encoding.Encode(ThinkingPlaceholder).Count;
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

    private static IEnumerable<object> EnumerateJsonArray(JsonElement element)
    {
        foreach (var item in element.EnumerateArray())
        {
            yield return item;
        }
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
}
