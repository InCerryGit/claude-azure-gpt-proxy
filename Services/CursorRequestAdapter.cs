using System.Text.Json;
using AzureGptProxy.Infrastructure;
using AzureGptProxy.Models;

namespace AzureGptProxy.Services;

public static class CursorRequestAdapter
{
    public static (JsonDocument Body, string InboundModel) BuildResponsesRequest(
        OpenAiChatCompletionsRequest request,
        NormalizedAzureOpenAiOptions azureOptions)
    {
        var inboundModel = request.Model?.Trim() ?? string.Empty;
        var effort = MapReasoningEffort(inboundModel);

        if (string.IsNullOrWhiteSpace(azureOptions.CursorAzureDeployment))
        {
            throw new InvalidOperationException("Missing CURSOR_AZURE_DEPLOYMENT.");
        }

        var (input, instructions) = MessagesToResponsesInputAndInstructions(request.Messages);

        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();

            writer.WriteString("model", azureOptions.CursorAzureDeployment);
            writer.WriteBoolean("stream", true);

            writer.WritePropertyName("input");
            input.WriteTo(writer);

            if (!string.IsNullOrWhiteSpace(instructions))
            {
                writer.WriteString("instructions", instructions);
            }

            if (request.Tools is { Count: > 0 })
            {
                writer.WritePropertyName("tools");
                WriteResponsesTools(writer, request.Tools);
            }

            if (request.ToolChoice is { ValueKind: not JsonValueKind.Undefined and not JsonValueKind.Null } toolChoice)
            {
                writer.WritePropertyName("tool_choice");
                toolChoice.WriteTo(writer);
            }

            if (!string.IsNullOrWhiteSpace(request.User))
            {
                writer.WriteString("prompt_cache_key", request.User);
            }

            writer.WritePropertyName("reasoning");
            writer.WriteStartObject();
            writer.WriteString("effort", effort);
            writer.WriteEndObject();

            writer.WriteEndObject();
        }

        stream.Position = 0;
        var doc = JsonDocument.Parse(stream);
        return (doc, inboundModel);
    }

    private static string MapReasoningEffort(string inboundModel)
    {
        // python: reasoning_effort = inbound_model.replace("gpt-", "").lower(); allow {high,medium,low,minimal}
        var effort = inboundModel.Replace("gpt-", "", StringComparison.OrdinalIgnoreCase).Trim().ToLowerInvariant();
        return effort switch
        {
            "high" => "high",
            "medium" => "medium",
            "low" => "low",
            "minimal" => "minimal",
            _ => throw new ArgumentException($"Invalid model '{inboundModel}'. Allowed: gpt-high|gpt-medium|gpt-low|gpt-minimal.")
        };
    }

    private static (JsonElement Input, string Instructions) MessagesToResponsesInputAndInstructions(List<OpenAiChatMessage> messages)
    {
        var instructionsParts = new List<string>();
        // Cursor in agent mode may send role=tool messages without tool_call_id.
        // If the upstream model cannot correlate tool outputs to prior function_call items,
        // it may repeat the same tool calls and trigger Cursor's "model looping" detector.
        // Track the pending tool call ids (in order) so we can infer call_id when missing.
        var pendingToolCallIds = new List<string>();

        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartArray();

            foreach (var m in messages)
            {
                var role = (m.Role ?? string.Empty).Trim();
                if (string.Equals(role, "system", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(role, "developer", StringComparison.OrdinalIgnoreCase))
                {
                    if (m.Content.ValueKind == JsonValueKind.String)
                    {
                        var s = m.Content.GetString();
                        if (!string.IsNullOrWhiteSpace(s))
                        {
                            instructionsParts.Add(s);
                        }
                    }
                    continue;
                }

                if (string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase))
                {
                    writer.WriteStartObject();
                    writer.WriteString("type", "function_call_output");
                    writer.WriteString("status", "completed");

                    var callId = m.ToolCallId;
                    if (string.IsNullOrWhiteSpace(callId) && pendingToolCallIds.Count > 0)
                    {
                        callId = pendingToolCallIds[0];
                        pendingToolCallIds.RemoveAt(0);
                    }
                    else if (!string.IsNullOrWhiteSpace(callId) && pendingToolCallIds.Count > 0)
                    {
                        // Keep pending list consistent even when tool_call_id is present.
                        var idx = pendingToolCallIds.FindIndex(id => string.Equals(id, callId, StringComparison.Ordinal));
                        if (idx >= 0)
                        {
                            pendingToolCallIds.RemoveAt(idx);
                        }
                    }

                    if (!string.IsNullOrWhiteSpace(callId))
                    {
                        writer.WriteString("call_id", callId);
                    }

                    writer.WritePropertyName("output");
                    // IMPORTANT: Azure Responses expects function_call_output.output to be a string (docs example),
                    // not an array of content parts. Cursor may send tool messages as an array like:
                    //   [{"type":"text","text":"..."}]
                    // If we forward that shape directly, Azure validates output[*].type and rejects "text".
                    // To preserve information without violating schema, serialize non-string content to JSON.
                    WriteToolOutputAsString(writer, m.Content);

                    writer.WriteEndObject();
                    continue;
                }

                // user / assistant
                writer.WriteStartObject();
                writer.WriteString("role", role.ToLowerInvariant());

                WriteResponsesContent(writer, role, m.Content);

                writer.WriteEndObject();

                // tool_calls on assistant messages -> Responses function_call items
                if (m.ToolCalls is { Count: > 0 })
                {
                    foreach (var tc in m.ToolCalls)
                    {
                        if (!string.IsNullOrWhiteSpace(tc.Id))
                        {
                            pendingToolCallIds.Add(tc.Id);
                        }

                        writer.WriteStartObject();
                        writer.WriteString("type", "function_call");
                        writer.WriteString("call_id", tc.Id);
                        writer.WriteString("name", tc.Function.Name);
                        writer.WritePropertyName("arguments");
                        writer.WriteStringValue(tc.Function.Arguments ?? string.Empty);
                        writer.WriteEndObject();
                    }
                }
            }

            writer.WriteEndArray();
        }

        stream.Position = 0;
        using var doc = JsonDocument.Parse(stream);
        return (doc.RootElement.Clone(), string.Join("\n\n", instructionsParts));
    }

    private static void WriteResponsesTools(Utf8JsonWriter writer, List<OpenAiTool> tools)
    {
        writer.WriteStartArray();
        foreach (var t in tools)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "function");
            writer.WriteString("name", t.Function.Name);

            if (!string.IsNullOrWhiteSpace(t.Function.Description))
            {
                writer.WriteString("description", t.Function.Description);
            }

            if (t.Function.Parameters is { ValueKind: not JsonValueKind.Undefined and not JsonValueKind.Null } parameters)
            {
                writer.WritePropertyName("parameters");
                parameters.WriteTo(writer);
            }

            writer.WriteBoolean("strict", false);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteResponsesContent(Utf8JsonWriter writer, string role, JsonElement content)
    {
        writer.WritePropertyName("content");
        writer.WriteStartArray();

        var wrotePart = false;
        void WriteContentPart(JsonElement part)
        {
            if (part.ValueKind != JsonValueKind.Object)
            {
                return;
            }

            var type = part.TryGetProperty("type", out var typeProp) ? typeProp.GetString() : null;

            if (string.Equals(type, "text", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "input_text", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "output_text", StringComparison.OrdinalIgnoreCase))
            {
                var text = part.TryGetProperty("text", out var textProp)
                    ? textProp.GetString() ?? string.Empty
                    : string.Empty;

                // Azure OpenAI Responses does NOT accept "text" as a content part type.
                // Additionally, for items inside the top-level "input" array, the content parts must be
                // "input_*" for role=user and "output_*" for role=assistant.
                // Normalize:
                // - any incoming "text" -> "input_text" for user, "output_text" for assistant
                // - any incoming "input_text" on assistant -> "output_text" (prevents 400 like: input[*].output[*].type='text')
                // - any incoming "output_text" on user -> "input_text" (keep symmetric)
                var isAssistant = string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase);
                var normalizedType = type?.ToLowerInvariant() switch
                {
                    "text" => isAssistant ? "output_text" : "input_text",
                    "input_text" => isAssistant ? "output_text" : "input_text",
                    "output_text" => isAssistant ? "output_text" : "input_text",
                    _ => isAssistant ? "output_text" : "input_text"
                };

                writer.WriteStartObject();
                writer.WriteString("type", normalizedType);
                writer.WriteString("text", text);
                writer.WriteEndObject();
                wrotePart = true;
                return;
            }

            if (string.Equals(type, "image_url", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "input_image", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "image", StringComparison.OrdinalIgnoreCase))
            {
                // Responses API expects input_image with image_url (string URL or data URL) or image_base64.
                if (TryWriteInputImage(writer, part))
                {
                    wrotePart = true;
                }
            }
        }

        if (content.ValueKind == JsonValueKind.Array)
        {
            foreach (var part in content.EnumerateArray())
            {
                WriteContentPart(part);
            }
        }
        else if (content.ValueKind == JsonValueKind.Object)
        {
            WriteContentPart(content);
        }

        if (!wrotePart)
        {
            // Fallback to plain text when content is a simple string or unsupported structure.
            var contentType = string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase) ? "output_text" : "input_text";
            writer.WriteStartObject();
            writer.WriteString("type", contentType);
            writer.WriteString("text", ExtractContentText(content));
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
    }

    private static void WriteImageUrl(Utf8JsonWriter writer, string? url)
    {
        if (string.IsNullOrWhiteSpace(url))
        {
            return;
        }

        writer.WriteString("image_url", url);
    }

    private static bool TryWriteInputImage(Utf8JsonWriter writer, JsonElement part)
    {
        // Ensure we never emit an empty input_image block.

        var url = ExtractImageUrlString(part);
        if (!string.IsNullOrWhiteSpace(url))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "input_image");
            writer.WriteString("image_url", url);
            writer.WriteEndObject();
            return true;
        }

        if (part.TryGetProperty("image_base64", out var imageBase64))
        {
            if (imageBase64.ValueKind == JsonValueKind.String && string.IsNullOrWhiteSpace(imageBase64.GetString()))
            {
                return false;
            }

            writer.WriteStartObject();
            writer.WriteString("type", "input_image");
            writer.WritePropertyName("image_base64");
            imageBase64.WriteTo(writer);
            writer.WriteEndObject();
            return true;
        }

        return false;
    }

    private static string? ExtractImageUrlString(JsonElement part)
    {
        // OpenAI shape: {"type":"image_url","image_url":"..."} or {"image_url":{"url":"...","detail":"..."}}
        if (part.TryGetProperty("image_url", out var imageUrl))
        {
            if (imageUrl.ValueKind == JsonValueKind.String)
            {
                return imageUrl.GetString();
            }

            if (imageUrl.ValueKind == JsonValueKind.Object &&
                imageUrl.TryGetProperty("url", out var urlValue) &&
                urlValue.ValueKind == JsonValueKind.String)
            {
                return urlValue.GetString();
            }
        }

        // Alternate shapes.
        if (part.TryGetProperty("url", out var urlProp) && urlProp.ValueKind == JsonValueKind.String)
        {
            return urlProp.GetString();
        }

        // Claude/Anthropic style: {"type":"image","source":{"type":"url","url":"..."}} or base64.
        if (part.TryGetProperty("source", out var source) && source.ValueKind == JsonValueKind.Object)
        {
            var sourceType = source.TryGetProperty("type", out var sourceTypeProp) && sourceTypeProp.ValueKind == JsonValueKind.String
                ? sourceTypeProp.GetString()
                : null;

            if (string.Equals(sourceType, "url", StringComparison.OrdinalIgnoreCase) &&
                source.TryGetProperty("url", out var srcUrl) &&
                srcUrl.ValueKind == JsonValueKind.String)
            {
                return srcUrl.GetString();
            }

            if (string.Equals(sourceType, "base64", StringComparison.OrdinalIgnoreCase) &&
                source.TryGetProperty("data", out var dataProp) &&
                dataProp.ValueKind == JsonValueKind.String)
            {
                var base64 = dataProp.GetString();
                if (string.IsNullOrWhiteSpace(base64))
                {
                    return null;
                }

                var mediaType = source.TryGetProperty("media_type", out var mediaTypeProp) && mediaTypeProp.ValueKind == JsonValueKind.String
                    ? (mediaTypeProp.GetString() ?? "image/png")
                    : "image/png";

                return $"data:{mediaType};base64,{base64}";
            }
        }

        return null;
    }

    private static string ExtractContentText(JsonElement content)
    {
        // Keep minimal: if content is string -> return.
        // If array/object -> return its raw JSON string (Cursor project is looser here; adapter in python normalizes richer content).
        return content.ValueKind switch
        {
            JsonValueKind.String => content.GetString() ?? string.Empty,
            JsonValueKind.Undefined => string.Empty,
            JsonValueKind.Null => string.Empty,
            _ => content.GetRawText()
        };
    }

    private static void WriteContentAsTextOrJson(Utf8JsonWriter writer, JsonElement content)
    {
        // For tool outputs: python forwards text; we keep string when possible.
        if (content.ValueKind == JsonValueKind.String)
        {
            writer.WriteStringValue(content.GetString() ?? string.Empty);
            return;
        }

        if (content.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            writer.WriteStringValue(string.Empty);
            return;
        }

        content.WriteTo(writer);
    }

    private static void WriteToolOutputAsString(Utf8JsonWriter writer, JsonElement content)
    {
        // Match old python behavior: function_call_output.output is the tool output content.
        // Azure Responses schema expects it as a string; if upstream provided structured JSON,
        // keep it by serializing to a JSON string.
        if (content.ValueKind == JsonValueKind.String)
        {
            writer.WriteStringValue(content.GetString() ?? string.Empty);
            return;
        }

        if (content.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            writer.WriteStringValue(string.Empty);
            return;
        }

        // Arrays/objects/numbers/bools: preserve as JSON text inside a string.
        writer.WriteStringValue(content.GetRawText());
    }
}
