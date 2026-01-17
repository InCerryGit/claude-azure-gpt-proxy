using System.Text.Json.Serialization;

namespace ClaudeAzureGptProxy.Models;

public sealed record ContentBlockText
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "text";

    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;
}

public sealed record ContentBlockImage
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "image";

    [JsonPropertyName("source")]
    public Dictionary<string, object?> Source { get; init; } = new();
}

public sealed record ContentBlockToolUse
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "tool_use";

    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("input")]
    public Dictionary<string, object?> Input { get; init; } = new();
}

public sealed record ContentBlockToolResult
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "tool_result";

    [JsonPropertyName("tool_use_id")]
    public string ToolUseId { get; init; } = string.Empty;

    [JsonPropertyName("content")]
    public object? Content { get; init; }
}

public sealed record SystemContent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "text";

    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;
}

public sealed record Message
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = "user";

    [JsonPropertyName("content")]
    public object? Content { get; init; }
}

public sealed record Tool
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("input_schema")]
    public Dictionary<string, object?> InputSchema { get; init; } = new();
}

public sealed record ToolChoice
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "auto";

    [JsonPropertyName("name")]
    public string? Name { get; init; }
}

public sealed record ThinkingConfig
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; init; } = true;
}

public sealed record MessagesRequest
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; init; }

    [JsonPropertyName("messages")]
    public List<Message> Messages { get; init; } = new();

    [JsonPropertyName("system")]
    public object? System { get; init; }

    [JsonPropertyName("stop_sequences")]
    public List<string>? StopSequences { get; init; }

    [JsonPropertyName("stream")]
    public bool Stream { get; init; }

    [JsonPropertyName("temperature")]
    public double? Temperature { get; init; } = 1.0;

    [JsonPropertyName("top_p")]
    public double? TopP { get; init; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, object?>? Metadata { get; init; }

    [JsonPropertyName("tools")]
    public List<Tool>? Tools { get; init; }

    [JsonPropertyName("tool_choice")]
    public ToolChoice? ToolChoice { get; init; }

    [JsonPropertyName("thinking")]
    public ThinkingConfig? Thinking { get; init; }

    [JsonIgnore]
    public string? OriginalModel { get; set; }

    [JsonIgnore]
    public string? ResolvedAzureModel { get; set; }
}

public sealed record TokenCountRequest
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("messages")]
    public List<Message> Messages { get; init; } = new();

    [JsonPropertyName("system")]
    public object? System { get; init; }

    [JsonPropertyName("tools")]
    public List<Tool>? Tools { get; init; }

    [JsonPropertyName("thinking")]
    public ThinkingConfig? Thinking { get; init; }

    [JsonPropertyName("tool_choice")]
    public ToolChoice? ToolChoice { get; init; }

    [JsonIgnore]
    public string? OriginalModel { get; set; }

    [JsonIgnore]
    public string? ResolvedAzureModel { get; set; }
}

public sealed record TokenCountResponse
{
    [JsonPropertyName("input_tokens")]
    public int InputTokens { get; init; }
}

public sealed record Usage
{
    [JsonPropertyName("input_tokens")]
    public int InputTokens { get; init; }

    [JsonPropertyName("output_tokens")]
    public int OutputTokens { get; init; }

    [JsonPropertyName("cache_creation_input_tokens")]
    public int CacheCreationInputTokens { get; init; }

    [JsonPropertyName("cache_read_input_tokens")]
    public int CacheReadInputTokens { get; init; }
}

public sealed record MessagesResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("role")]
    public string Role { get; init; } = "assistant";

    [JsonPropertyName("content")]
    public List<Dictionary<string, object?>> Content { get; init; } = new();

    [JsonPropertyName("type")]
    public string Type { get; init; } = "message";

    [JsonPropertyName("stop_reason")]
    public string? StopReason { get; init; }

    [JsonPropertyName("stop_sequence")]
    public string? StopSequence { get; init; }

    [JsonPropertyName("usage")]
    public Usage Usage { get; init; } = new();
}
