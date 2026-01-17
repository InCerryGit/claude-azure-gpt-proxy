using System.Linq;
using System.Text.Json;
using ClaudeAzureGptProxy.Models;
using Microsoft.Extensions.Logging;
using OpenAI.Chat;

namespace ClaudeAzureGptProxy.Services;

public static class SseStreaming
{
    public sealed class StreamStats
    {
        public int ChunkCount { get; set; }
        public int EventCount { get; set; }
        public int OutputCharacters { get; set; }
        public int InputTokens { get; set; }
        public int OutputTokens { get; set; }
        public string? StopReason { get; set; }
    }

    private static string EmitEvent(string eventType, object payload)
    {
        return $"event: {eventType}\ndata: {JsonSerializer.Serialize(payload)}\n\n";
    }

    private static string EmitMessageStart(string messageId, string responseModel, int inputTokens, int outputTokens)
    {
        var messageData = new
        {
            type = "message_start",
            message = new
            {
                id = messageId,
                type = "message",
                role = "assistant",
                model = responseModel,
                content = Array.Empty<object>(),
                stop_reason = (string?)null,
                stop_sequence = (string?)null,
                usage = new
                {
                    input_tokens = inputTokens,
                    cache_creation_input_tokens = 0,
                    cache_read_input_tokens = 0,
                    output_tokens = outputTokens
                }
            }
        };

        return EmitEvent("message_start", messageData);
    }

    private static string EmitContentBlockStart(int index, object contentBlock)
    {
        return EmitEvent("content_block_start", new
        {
            type = "content_block_start",
            index,
            content_block = contentBlock
        });
    }

    private static string EmitContentBlockDelta(int index, object delta)
    {
        return EmitEvent("content_block_delta", new
        {
            type = "content_block_delta",
            index,
            delta
        });
    }

    private static string EmitContentBlockStop(int index)
    {
        return EmitEvent("content_block_stop", new
        {
            type = "content_block_stop",
            index
        });
    }

    private static string EmitMessageDelta(string stopReason, int outputTokens)
    {
        return EmitEvent("message_delta", new
        {
            type = "message_delta",
            delta = new { stop_reason = stopReason, stop_sequence = (string?)null },
            usage = new { output_tokens = outputTokens }
        });
    }

    private static string EmitMessageStop()
    {
        return EmitEvent("message_stop", new { type = "message_stop" });
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

    public static async IAsyncEnumerable<string> HandleStreaming(
        IAsyncEnumerable<Dictionary<string, object?>> responseStream,
        MessagesRequest originalRequest,
        ILogger logger,
        StreamStats? stats = null)
    {
        var messageId = $"msg_{Guid.NewGuid():N}";
        var responseModel = originalRequest.OriginalModel ?? originalRequest.Model;

        logger.LogInformation("Streaming start messageId={MessageId} model={Model}", messageId, responseModel);

        stats ??= new StreamStats();
        // Match old/server.py behavior: for synthesized streams (e.g. Azure Responses bridged to a single chunk)
        // we already know usage before sending message_start; for regular streams usage is typically unknown.
        await using var enumerator = responseStream.GetAsyncEnumerator();

        int? toolIndex = null;
        var accumulatedText = string.Empty;
        var textSent = false;
        var textBlockClosed = false;
        var inputTokens = 0;
        var outputTokens = 0;
        var hasSentStopReason = false;
        var lastToolIndex = 0;

        if (!await enumerator.MoveNextAsync())
        {
            yield return EmitMessageStart(messageId, responseModel, 0, 0);
            yield return EmitContentBlockStart(0, new { type = "text", text = string.Empty });
            yield return EmitEvent("ping", new { type = "ping" });
            yield return EmitMessageDelta("end_turn", 0);
            yield return EmitMessageStop();
            yield return "data: [DONE]\n\n";

            stats.EventCount += 6;
            stats.StopReason = "end_turn";
            logger.LogInformation(
                "Streaming ended without chunks messageId={MessageId} chunks=0 events={EventCount}",
                messageId,
                stats.EventCount);
            yield break;
        }

        var firstChunk = enumerator.Current;
        UpdateUsage(firstChunk, ref inputTokens, ref outputTokens);

        yield return EmitMessageStart(messageId, responseModel, inputTokens, outputTokens);
        yield return EmitContentBlockStart(0, new { type = "text", text = string.Empty });
        yield return EmitEvent("ping", new { type = "ping" });

        stats.EventCount += 3;

        // Process the first chunk we already pulled, then continue with the remaining stream.
        foreach (var chunk in new[] { firstChunk })
        {
            stats.ChunkCount += 1;

            List<string> events;
            try
            {
                events = ProcessStreamChunk(
                    chunk,
                    ref toolIndex,
                    ref accumulatedText,
                    ref textSent,
                    ref textBlockClosed,
                    ref inputTokens,
                    ref outputTokens,
                    ref hasSentStopReason,
                    ref lastToolIndex,
                    stats);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error processing stream chunk messageId={MessageId} chunkCount={ChunkCount}", messageId, stats.ChunkCount);
                events = new List<string>();
            }

            stats.EventCount += events.Count;
            foreach (var evt in events)
            {
                yield return evt;
            }

            if (hasSentStopReason)
            {
                logger.LogInformation(
                    "Streaming completed messageId={MessageId} stopReason={StopReason} inputTokens={InputTokens} outputTokens={OutputTokens} chunks={ChunkCount} events={EventCount}",
                    messageId,
                    stats.StopReason ?? "(unknown)",
                    stats.InputTokens,
                    stats.OutputTokens,
                    stats.ChunkCount,
                    stats.EventCount);
                yield break;
            }
        }

        while (await enumerator.MoveNextAsync())
        {
            var chunk = enumerator.Current;
            stats.ChunkCount += 1;

            List<string> events;
            try
            {
                events = ProcessStreamChunk(
                    chunk,
                    ref toolIndex,
                    ref accumulatedText,
                    ref textSent,
                    ref textBlockClosed,
                    ref inputTokens,
                    ref outputTokens,
                    ref hasSentStopReason,
                    ref lastToolIndex,
                    stats);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error processing stream chunk messageId={MessageId} chunkCount={ChunkCount}", messageId, stats.ChunkCount);
                continue;
            }

            stats.EventCount += events.Count;
            foreach (var evt in events)
            {
                yield return evt;
            }

            if (hasSentStopReason)
            {
                logger.LogInformation(
                    "Streaming completed messageId={MessageId} stopReason={StopReason} inputTokens={InputTokens} outputTokens={OutputTokens} chunks={ChunkCount} events={EventCount}",
                    messageId,
                    stats.StopReason ?? "(unknown)",
                    stats.InputTokens,
                    stats.OutputTokens,
                    stats.ChunkCount,
                    stats.EventCount);
                yield break;
            }
        }

        if (!hasSentStopReason)
        {
            var remainingEvents = CloseOpenBlocks(toolIndex, lastToolIndex, textBlockClosed, accumulatedText, textSent).ToList();
            foreach (var evt in remainingEvents)
            {
                yield return evt;
            }

            yield return EmitMessageDelta("end_turn", outputTokens);
            yield return EmitMessageStop();
            yield return "data: [DONE]\n\n";

            stats.EventCount += remainingEvents.Count + 3;
            stats.OutputCharacters = accumulatedText.Length;
            stats.InputTokens = inputTokens;
            stats.OutputTokens = outputTokens;
            stats.StopReason = "end_turn";

            logger.LogInformation(
                "Streaming ended without finish_reason messageId={MessageId} inputTokens={InputTokens} outputTokens={OutputTokens} chunks={ChunkCount} events={EventCount}",
                messageId,
                stats.InputTokens,
                stats.OutputTokens,
                stats.ChunkCount,
                stats.EventCount);
        }
    }

    public static async IAsyncEnumerable<string> HandleSynthStream(MessagesResponse response)
    {
        var messageId = response.Id;
        var responseModel = response.Model;

        var messageData = new
        {
            type = "message_start",
            message = new
            {
                id = messageId,
                type = "message",
                role = "assistant",
                model = responseModel,
                content = Array.Empty<object>(),
                stop_reason = (string?)null,
                stop_sequence = (string?)null,
                usage = new
                {
                    input_tokens = response.Usage.InputTokens,
                    cache_creation_input_tokens = 0,
                    cache_read_input_tokens = 0,
                    output_tokens = response.Usage.OutputTokens
                }
            }
        };

        yield return EmitEvent("message_start", messageData);

        var contentBlocks = response.Content ?? new List<Dictionary<string, object?>>();
        var blockIndex = 0;

        foreach (var block in contentBlocks)
        {
            if (!block.TryGetValue("type", out var typeObj))
            {
                continue;
            }

            var blockType = typeObj?.ToString();
            if (blockType == "text")
            {
                var text = block.TryGetValue("text", out var textObj) ? textObj?.ToString() ?? string.Empty : string.Empty;
                yield return EmitContentBlockStart(blockIndex, new { type = "text", text = string.Empty });
                if (!string.IsNullOrEmpty(text))
                {
                    yield return EmitContentBlockDelta(blockIndex, new { type = "text_delta", text });
                }
                yield return EmitContentBlockStop(blockIndex);
            }
            else if (blockType == "tool_use")
            {
                yield return EmitContentBlockStart(blockIndex, block);
                yield return EmitContentBlockStop(blockIndex);
            }

            blockIndex++;
        }

        yield return EmitMessageDelta(response.StopReason ?? "end_turn", response.Usage.OutputTokens);
        yield return EmitMessageStop();
        yield return "data: [DONE]\n\n";
    }

    private static void UpdateUsage(Dictionary<string, object?> chunk, ref int inputTokens, ref int outputTokens)
    {
        if (!chunk.TryGetValue("usage", out var usageObj) || usageObj is null)
        {
            return;
        }

        if (usageObj is JsonElement usageElement && usageElement.ValueKind == JsonValueKind.Object)
        {
            if (usageElement.TryGetProperty("prompt_tokens", out var promptProp))
            {
                inputTokens = promptProp.GetInt32();
            }

            if (usageElement.TryGetProperty("completion_tokens", out var completionProp))
            {
                outputTokens = completionProp.GetInt32();
            }
            return;
        }

        if (usageObj is IDictionary<string, object?> usageDict)
        {
            if (usageDict.TryGetValue("prompt_tokens", out var promptVal) &&
                int.TryParse(promptVal?.ToString(), out var promptParsed))
            {
                inputTokens = promptParsed;
            }

            if (usageDict.TryGetValue("completion_tokens", out var completionVal) &&
                int.TryParse(completionVal?.ToString(), out var completionParsed))
            {
                outputTokens = completionParsed;
            }
        }
    }

    private static (object? delta, string? finishReason) GetDeltaPayload(Dictionary<string, object?> chunk)
    {
        if (chunk.TryGetValue("choices", out var choicesObj))
        {
            if (choicesObj is JsonElement choicesElement && choicesElement.ValueKind == JsonValueKind.Array &&
                choicesElement.GetArrayLength() > 0)
            {
                var choice = choicesElement[0];
                var delta = choice.TryGetProperty("delta", out var deltaProp)
                    ? deltaProp
                    : choice.TryGetProperty("message", out var messageProp) ? messageProp : default(JsonElement);
                var finishReason = choice.TryGetProperty("finish_reason", out var finishProp)
                    ? finishProp.GetString()
                    : null;
                return (delta.ValueKind == JsonValueKind.Undefined ? null : delta, finishReason);
            }

            if (choicesObj is IEnumerable<object> choicesList)
            {
                var choice = choicesList.FirstOrDefault();
                if (choice is IDictionary<string, object?> dict)
                {
                    var delta = dict.TryGetValue("delta", out var deltaObj) ? deltaObj :
                        dict.TryGetValue("message", out var msgObj) ? msgObj : null;
                    var finishReason = dict.TryGetValue("finish_reason", out var finishObj)
                        ? finishObj?.ToString()
                        : null;
                    return (delta, finishReason);
                }
            }
        }

        return (null, null);
    }

    private static string? ExtractDeltaContent(object delta)
    {
        if (delta is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Object)
            {
                if (element.TryGetProperty("content", out var contentProp))
                {
                    return contentProp.GetString();
                }

                if (element.TryGetProperty("text", out var textProp))
                {
                    return textProp.GetString();
                }
            }

            if (element.ValueKind == JsonValueKind.Array)
            {
                var contentBuilder = new System.Text.StringBuilder();
                foreach (var item in element.EnumerateArray())
                {
                    if (item.ValueKind == JsonValueKind.Object &&
                        item.TryGetProperty("text", out var textProp))
                    {
                        contentBuilder.Append(textProp.GetString());
                    }
                }

                var combined = contentBuilder.ToString();
                return string.IsNullOrEmpty(combined) ? null : combined;
            }
        }

        if (delta is IDictionary<string, object?> dict)
        {
            if (dict.TryGetValue("content", out var contentObj))
            {
                return contentObj?.ToString();
            }

            if (dict.TryGetValue("text", out var textObj))
            {
                return textObj?.ToString();
            }
        }

        return null;
    }

    private static object? ExtractDeltaToolCalls(object delta)
    {
        if (delta is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty("tool_calls", out var toolProp))
            {
                return toolProp;
            }
        }

        if (delta is IDictionary<string, object?> dict && dict.TryGetValue("tool_calls", out var toolObj))
        {
            return toolObj;
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

    private static List<string> ProcessStreamChunk(
        Dictionary<string, object?> chunk,
        ref int? toolIndex,
        ref string accumulatedText,
        ref bool textSent,
        ref bool textBlockClosed,
        ref int inputTokens,
        ref int outputTokens,
        ref bool hasSentStopReason,
        ref int lastToolIndex,
        StreamStats stats)
    {
        var events = new List<string>();

        UpdateUsage(chunk, ref inputTokens, ref outputTokens);
        stats.InputTokens = inputTokens;
        stats.OutputTokens = outputTokens;

        var (delta, finishReason) = GetDeltaPayload(chunk);
        if (delta is null)
        {
            return events;
        }

        var deltaContent = ExtractDeltaContent(delta);
        var deltaToolCalls = ExtractDeltaToolCalls(delta);

        if (!string.IsNullOrEmpty(deltaContent))
        {
            accumulatedText += deltaContent;
            stats.OutputCharacters = accumulatedText.Length;
            if (toolIndex is null && !textBlockClosed)
            {
                textSent = true;
                events.Add(EmitContentBlockDelta(0, new { type = "text_delta", text = deltaContent }));
            }
        }

        if (deltaToolCalls is not null)
        {
            if (toolIndex is null)
            {
                foreach (var evt in CloseTextBlockIfNeeded(
                             toolIndex, textBlockClosed, accumulatedText, textSent))
                {
                    if (evt is not null)
                    {
                        textBlockClosed = true;
                        events.Add(evt);
                    }
                }
            }

            IEnumerable<object> toolCalls = deltaToolCalls switch
            {
                JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
                IEnumerable<object> list => list,
                _ => new[] { deltaToolCalls }
            };

            foreach (var toolCall in toolCalls)
            {
                var (newToolIndex, newLastToolIndex, createdEvents) = HandleToolDelta(
                    toolCall, toolIndex, lastToolIndex);
                toolIndex = newToolIndex;
                lastToolIndex = newLastToolIndex;
                events.AddRange(createdEvents);
            }
        }

        if (!string.IsNullOrWhiteSpace(finishReason) && !hasSentStopReason)
        {
            hasSentStopReason = true;
            events.AddRange(CloseOpenBlocks(toolIndex, lastToolIndex, textBlockClosed, accumulatedText, textSent));

            var stopReason = MapStopReason(finishReason);
            stats.StopReason = stopReason;
            events.Add(EmitMessageDelta(stopReason, outputTokens));
            events.Add(EmitMessageStop());
            events.Add("data: [DONE]\n\n");
        }

        return events;
    }

    private static IEnumerable<string> CloseOpenBlocks(
        int? toolIndex,
        int lastToolIndex,
        bool textBlockClosed,
        string accumulatedText,
        bool textSent)
    {
        if (toolIndex is not null)
        {
            for (var i = 1; i <= lastToolIndex; i++)
            {
                yield return EmitContentBlockStop(i);
            }
        }

        if (!textBlockClosed)
        {
            if (!string.IsNullOrEmpty(accumulatedText) && !textSent)
            {
                yield return EmitContentBlockDelta(0, new { type = "text_delta", text = accumulatedText });
            }

            yield return EmitContentBlockStop(0);
        }
    }

    private static IEnumerable<string?> CloseTextBlockIfNeeded(
        int? toolIndex,
        bool textBlockClosed,
        string accumulatedText,
        bool textSent)
    {
        if (toolIndex is null && !textBlockClosed)
        {
            if (textSent)
            {
                yield return EmitContentBlockStop(0);
            }
            else if (!string.IsNullOrEmpty(accumulatedText))
            {
                yield return EmitContentBlockDelta(0, new { type = "text_delta", text = accumulatedText });
                yield return EmitContentBlockStop(0);
            }
            else
            {
                yield return EmitContentBlockStop(0);
            }
        }

        yield return null;
    }

    private static (int? toolIndex, int lastToolIndex, List<string> events) HandleToolDelta(
        object toolCall,
        int? toolIndex,
        int lastToolIndex)
    {
        var currentIndex = 0;
        if (toolCall is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty("index", out var indexProp) && indexProp.ValueKind == JsonValueKind.Number)
            {
                currentIndex = indexProp.GetInt32();
            }
        }
        else if (toolCall is IDictionary<string, object?> dict &&
                 dict.TryGetValue("index", out var indexObj) &&
                 int.TryParse(indexObj?.ToString(), out var parsed))
        {
            currentIndex = parsed;
        }
        else if (toolCall is StreamingChatToolCallUpdate update)
        {
            currentIndex = update.Index;
        }

        var createdEvents = new List<string>();
        int? anthropicToolIndex = null;

        if (toolIndex is null || currentIndex != toolIndex)
        {
            toolIndex = currentIndex;
            lastToolIndex += 1;
            anthropicToolIndex = lastToolIndex;

            var function = ExtractField(toolCall, "function");
            var name = ExtractString(function, "name") ?? string.Empty;
            var toolId = ExtractString(toolCall, "id") ?? $"toolu_{Guid.NewGuid():N}";

            createdEvents.Add(EmitContentBlockStart(anthropicToolIndex.Value, new
            {
                type = "tool_use",
                id = toolId,
                name,
                input = new Dictionary<string, object?>()
            }));
        }

        var argumentsRaw = ExtractField(ExtractField(toolCall, "function"), "arguments");
        if (argumentsRaw is not null)
        {
            var argsJson = NormalizeArguments(argumentsRaw);
            if (anthropicToolIndex is not null)
            {
                createdEvents.Add(EmitContentBlockDelta(anthropicToolIndex.Value, new
                {
                    type = "input_json_delta",
                    partial_json = argsJson
                }));
            }
        }

        return (toolIndex, lastToolIndex, createdEvents);
    }

    private static object? ExtractField(object? obj, string name)
    {
        if (obj is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty(name, out var prop))
            {
                return prop;
            }
        }

        if (obj is IDictionary<string, object?> dict && dict.TryGetValue(name, out var value))
        {
            return value;
        }

        if (obj is StreamingChatToolCallUpdate update)
        {
            return name switch
            {
                "function" => update,
                "arguments" => update.FunctionArgumentsUpdate,
                "id" => update.ToolCallId,
                "index" => update.Index,
                _ => null
            };
        }

        if (obj is ChatToolCall toolCall)
        {
            return name switch
            {
                "function" => toolCall,
                "arguments" => toolCall.FunctionArguments,
                "id" => toolCall.Id,
                _ => null
            };
        }

        return null;
    }

    private static string? ExtractString(object? obj, string name)
    {
        if (obj is JsonElement element && element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(name, out var prop))
        {
            return prop.GetString();
        }

        if (obj is IDictionary<string, object?> dict && dict.TryGetValue(name, out var value))
        {
            return value?.ToString();
        }

        if (obj is StreamingChatToolCallUpdate update)
        {
            return name switch
            {
                "name" => update.FunctionName,
                "id" => update.ToolCallId,
                _ => null
            };
        }

        if (obj is ChatToolCall toolCall)
        {
            return name switch
            {
                "name" => toolCall.FunctionName,
                "id" => toolCall.Id,
                _ => null
            };
        }

        return null;
    }

    private static string NormalizeArguments(object arguments)
    {
        if (arguments is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String)
            {
                return element.GetString() ?? string.Empty;
            }

            return element.GetRawText();
        }

        if (arguments is string text)
        {
            return text;
        }

        if (arguments is StreamingChatToolCallUpdate update)
        {
            return update.FunctionArgumentsUpdate?.ToString() ?? string.Empty;
        }

        if (arguments is ChatToolCall toolCall)
        {
            return toolCall.FunctionArguments?.ToString() ?? string.Empty;
        }

        if (arguments is BinaryData binaryData)
        {
            return binaryData.ToString();
        }

        return JsonSerializer.Serialize(arguments);
    }
}
