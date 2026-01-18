using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using AzureGptProxy.Models;
using Microsoft.Extensions.Logging;
using OpenAI.Chat;

namespace AzureGptProxy.Services;

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

        public MessagesResponse? AggregatedResponse { get; set; }
    }

    private sealed class ToolUseAggregate
    {
        public required string Id { get; init; }
        public required string Name { get; init; }
        public StringBuilder Arguments { get; } = new();
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    private static string EmitEvent(string eventType, object payload)
    {
        return $"event: {eventType}\ndata: {JsonSerializer.Serialize(payload, JsonOptions)}\n\n";
    }

    private static MessagesResponse BuildAggregatedResponse(
        string messageId,
        string responseModel,
        string accumulatedText,
        Dictionary<string, int> toolKeyToAnthropicIndex,
        Dictionary<int, ToolUseAggregate> toolAggregates,
        int inputTokens,
        int outputTokens,
        int cacheReadInputTokens,
        int cacheCreationInputTokens,
        string? stopReason)
    {
        var content = new List<Dictionary<string, object?>>();

        // text block index=0 is always present in this implementation
        content.Add(new Dictionary<string, object?>
        {
            ["type"] = "text",
            ["text"] = accumulatedText
        });

        // We can determine how many tool blocks were started by taking the max mapped index.
        // Tool content blocks are 1..N.
        var maxToolIndex = toolKeyToAnthropicIndex.Count == 0 ? 0 : toolKeyToAnthropicIndex.Values.Max();
        for (var i = 1; i <= maxToolIndex; i++)
        {
            if (toolAggregates.TryGetValue(i, out var aggregate))
            {
                var input = ParseToolInputJson(aggregate.Arguments.ToString());
                content.Add(new Dictionary<string, object?>
                {
                    ["type"] = "tool_use",
                    ["id"] = aggregate.Id,
                    ["name"] = aggregate.Name,
                    ["input"] = input
                });
            }
            else
            {
                content.Add(new Dictionary<string, object?>
                {
                    ["type"] = "tool_use",
                    ["id"] = string.Empty,
                    ["name"] = string.Empty,
                    ["input"] = new Dictionary<string, object?>()
                });
            }
        }

        return new MessagesResponse
        {
            Id = messageId,
            Model = responseModel,
            Role = "assistant",
            Content = content,
            StopReason = stopReason,
            StopSequence = null,
            Usage = new Usage
            {
                InputTokens = inputTokens,
                OutputTokens = outputTokens,
                CacheCreationInputTokens = cacheCreationInputTokens,
                CacheReadInputTokens = cacheReadInputTokens
            }
        };
    }


    private static string EmitMessageStart(
        string messageId,
        string responseModel,
        int inputTokens,
        int outputTokens,
        int cacheReadInputTokens,
        int cacheCreationInputTokens)
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
                    cache_creation_input_tokens = cacheCreationInputTokens,
                    cache_read_input_tokens = cacheReadInputTokens,
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

    private static string EmitMessageDelta(
        string stopReason,
        int inputTokens,
        int outputTokens,
        int cacheReadInputTokens,
        int cacheCreationInputTokens)
    {
        return EmitEvent("message_delta", new
        {
            type = "message_delta",
            delta = new { stop_reason = stopReason, stop_sequence = (string?)null },
            usage = new
            {
                input_tokens = inputTokens,
                cache_creation_input_tokens = cacheCreationInputTokens,
                cache_read_input_tokens = cacheReadInputTokens,
                output_tokens = outputTokens
            }
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
            "stop" => "end_turn",
            "content_filter" => "content_filter",
            "cancelled" => "cancelled",
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
        var responseModel = AnthropicConversion.StripProviderPrefix(originalRequest.OriginalModel ?? originalRequest.Model);

        logger.LogInformation("Streaming start messageId={MessageId} model={Model}", messageId, responseModel);

        stats ??= new StreamStats();
        // Match old/server.py behavior: for synthesized streams (e.g. Azure Responses bridged to a single chunk)
        // we already know usage before sending message_start; for regular streams usage is typically unknown.
        await using var enumerator = responseStream.GetAsyncEnumerator();

        // Per-stream mapping: OpenAI tool call (id preferred, else index) -> Anthropic content block index.
        // This must be per request; any shared/static map will break concurrent streams.
        var toolKeyToAnthropicIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var toolAggregates = new Dictionary<int, ToolUseAggregate>();
        var unknownToolKeySeq = 0;
        int? toolIndex = null;
        var accumulatedText = string.Empty;
        var textSent = false;
        var textBlockClosed = false;
        var inputTokens = 0;
        var outputTokens = 0;
        var cacheReadInputTokens = 0;
        var cacheCreationInputTokens = 0;
        var hasSentStopReason = false;
        var lastToolIndex = 0;

        if (!await enumerator.MoveNextAsync())
        {
            yield return EmitMessageStart(messageId, responseModel, 0, 0, 0, 0);
            yield return EmitContentBlockStart(0, new { type = "text", text = string.Empty });
            yield return EmitEvent("ping", new { type = "ping" });
            yield return EmitMessageDelta("end_turn", 0, 0, 0, 0);
            yield return EmitMessageStop();
            yield return "data: [DONE]\n\n";

            stats.EventCount += 6;
            stats.StopReason = "end_turn";
            stats.AggregatedResponse = new MessagesResponse
            {
                Id = messageId,
                Model = responseModel,
                Role = "assistant",
                Content = new List<Dictionary<string, object?>>
                {
                    new()
                    {
                        ["type"] = "text",
                        ["text"] = string.Empty
                    }
                },
                StopReason = "end_turn",
                StopSequence = null,
                Usage = new Usage
                {
                    InputTokens = 0,
                    OutputTokens = 0,
                    CacheCreationInputTokens = 0,
                    CacheReadInputTokens = 0
                }
            };
            logger.LogInformation(
                "Streaming ended without chunks messageId={MessageId} chunks=0 events={EventCount}",
                messageId,
                stats.EventCount);
            yield break;
        }

        var firstChunk = enumerator.Current;
        UpdateUsage(firstChunk, ref inputTokens, ref outputTokens, ref cacheReadInputTokens);

        yield return EmitMessageStart(
            messageId,
            responseModel,
            inputTokens,
            outputTokens,
            cacheReadInputTokens,
            cacheCreationInputTokens);
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
                    messageId,
                    responseModel,
                    chunk,
                    ref toolIndex,
                    ref accumulatedText,
                    ref textSent,
                    ref textBlockClosed,
                    ref inputTokens,
                    ref outputTokens,
                    ref cacheReadInputTokens,
                    cacheCreationInputTokens,
                    ref hasSentStopReason,
                    ref lastToolIndex,
                    toolKeyToAnthropicIndex,
                    toolAggregates,
                    ref unknownToolKeySeq,
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
                    messageId,
                    responseModel,
                    chunk,
                    ref toolIndex,
                    ref accumulatedText,
                    ref textSent,
                    ref textBlockClosed,
                    ref inputTokens,
                    ref outputTokens,
                    ref cacheReadInputTokens,
                    cacheCreationInputTokens,
                    ref hasSentStopReason,
                    ref lastToolIndex,
                    toolKeyToAnthropicIndex,
                    toolAggregates,
                    ref unknownToolKeySeq,
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

            yield return EmitMessageDelta("end_turn", inputTokens, outputTokens, cacheReadInputTokens, cacheCreationInputTokens);
            yield return EmitMessageStop();
            yield return "data: [DONE]\n\n";

            stats.EventCount += remainingEvents.Count + 3;
            stats.OutputCharacters = accumulatedText.Length;
            stats.InputTokens = inputTokens;
            stats.OutputTokens = outputTokens;
            stats.StopReason = "end_turn";
            stats.AggregatedResponse = BuildAggregatedResponse(
                messageId,
                responseModel,
                accumulatedText,
                toolKeyToAnthropicIndex,
                toolAggregates,
                inputTokens,
                outputTokens,
                cacheReadInputTokens,
                cacheCreationInputTokens,
                stats.StopReason);

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
                    cache_creation_input_tokens = response.Usage.CacheCreationInputTokens,
                    cache_read_input_tokens = response.Usage.CacheReadInputTokens,
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

        yield return EmitMessageDelta(
            response.StopReason ?? "end_turn",
            response.Usage.InputTokens,
            response.Usage.OutputTokens,
            response.Usage.CacheReadInputTokens,
            response.Usage.CacheCreationInputTokens);
        yield return EmitMessageStop();
        yield return "data: [DONE]\n\n";
    }

    private static void UpdateUsage(
        Dictionary<string, object?> chunk,
        ref int inputTokens,
        ref int outputTokens,
        ref int cacheReadInputTokens)
    {
        if (!chunk.TryGetValue("usage", out var usageObj) || usageObj is null)
        {
            return;
        }

        if (usageObj is JsonElement usageElement && usageElement.ValueKind == JsonValueKind.Object)
        {
            // Chat Completions: {prompt_tokens, completion_tokens}
            // Responses: {input_tokens, output_tokens}
            if (usageElement.TryGetProperty("prompt_tokens", out var promptProp) ||
                usageElement.TryGetProperty("input_tokens", out promptProp))
            {
                if (promptProp.ValueKind == JsonValueKind.Number)
                {
                    inputTokens = promptProp.GetInt32();
                }
            }

            if (usageElement.TryGetProperty("completion_tokens", out var completionProp) ||
                usageElement.TryGetProperty("output_tokens", out completionProp))
            {
                if (completionProp.ValueKind == JsonValueKind.Number)
                {
                    outputTokens = completionProp.GetInt32();
                }
            }

            if (usageElement.TryGetProperty("input_tokens_details", out var inputDetails) &&
                inputDetails.ValueKind == JsonValueKind.Object &&
                inputDetails.TryGetProperty("cached_tokens", out var cachedTokensProp) &&
                cachedTokensProp.ValueKind == JsonValueKind.Number)
            {
                cacheReadInputTokens = cachedTokensProp.GetInt32();
            }
            return;
        }

        if (usageObj is IDictionary<string, object?> usageDict)
        {
            if ((usageDict.TryGetValue("prompt_tokens", out var promptVal) ||
                 usageDict.TryGetValue("input_tokens", out promptVal)) &&
                int.TryParse(promptVal?.ToString(), out var promptParsed))
            {
                inputTokens = promptParsed;
            }

            if ((usageDict.TryGetValue("completion_tokens", out var completionVal) ||
                 usageDict.TryGetValue("output_tokens", out completionVal)) &&
                int.TryParse(completionVal?.ToString(), out var completionParsed))
            {
                outputTokens = completionParsed;
            }

            if (usageDict.TryGetValue("input_tokens_details", out var inputDetailsObj) && inputDetailsObj is not null)
            {
                if (inputDetailsObj is JsonElement detailsEl && detailsEl.ValueKind == JsonValueKind.Object &&
                    detailsEl.TryGetProperty("cached_tokens", out var cachedTokensProp) &&
                    cachedTokensProp.ValueKind == JsonValueKind.Number)
                {
                    cacheReadInputTokens = cachedTokensProp.GetInt32();
                }
                else if (inputDetailsObj is IDictionary<string, object?> detailsDict &&
                         detailsDict.TryGetValue("cached_tokens", out var cachedTokensObj) &&
                         int.TryParse(cachedTokensObj?.ToString(), out var cachedTokens))
                {
                    cacheReadInputTokens = cachedTokens;
                }
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
        string messageId,
        string responseModel,
        Dictionary<string, object?> chunk,
        ref int? toolIndex,
        ref string accumulatedText,
        ref bool textSent,
        ref bool textBlockClosed,
        ref int inputTokens,
        ref int outputTokens,
        ref int cacheReadInputTokens,
        int cacheCreationInputTokens,
        ref bool hasSentStopReason,
        ref int lastToolIndex,
        Dictionary<string, int> toolKeyToAnthropicIndex,
        Dictionary<int, ToolUseAggregate> toolAggregates,
        ref int unknownToolKeySeq,
        StreamStats stats)
    {
        var events = new List<string>();

        UpdateUsage(chunk, ref inputTokens, ref outputTokens, ref cacheReadInputTokens);
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
                    toolCall, toolIndex, lastToolIndex, toolKeyToAnthropicIndex, toolAggregates, ref unknownToolKeySeq);
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
            stats.AggregatedResponse = BuildAggregatedResponse(
                messageId,
                responseModel,
                accumulatedText,
                toolKeyToAnthropicIndex,
                toolAggregates,
                inputTokens,
                outputTokens,
                cacheReadInputTokens,
                cacheCreationInputTokens,
                stopReason);
            events.Add(EmitMessageDelta(stopReason, inputTokens, outputTokens, cacheReadInputTokens, cacheCreationInputTokens));
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
        int lastToolIndex,
        Dictionary<string, int> toolKeyToAnthropicIndex,
        Dictionary<int, ToolUseAggregate> toolAggregates,
        ref int unknownToolKeySeq)
    {
        var toolId = ExtractString(toolCall, "id");

        var hasIndex = false;
        var currentIndex = 0;
        if (toolCall is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty("index", out var indexProp) && indexProp.ValueKind == JsonValueKind.Number)
            {
                currentIndex = indexProp.GetInt32();
                hasIndex = true;
            }
        }
        else if (toolCall is IDictionary<string, object?> dict &&
                 dict.TryGetValue("index", out var indexObj) &&
                 int.TryParse(indexObj?.ToString(), out var parsed))
        {
            currentIndex = parsed;
            hasIndex = true;
        }
        else if (toolCall is StreamingChatToolCallUpdate update)
        {
            currentIndex = update.Index;
            hasIndex = true;
        }

        var toolKey = !string.IsNullOrWhiteSpace(toolId)
            ? $"id:{toolId}"
            : hasIndex
                ? $"index:{currentIndex}"
                : $"unknown:{++unknownToolKeySeq}";

        var createdEvents = new List<string>();

        // Anthropic expects that tool_use input_json_delta uses the same content block index
        // across multiple chunks. We map each OpenAI tool call index -> Anthropic content block index.
        // This avoids only emitting the first partial args chunk.
        if (toolKeyToAnthropicIndex.TryGetValue(toolKey, out var existingAnthropicIndex))
        {
            // Any tool delta means we're now in tool mode (value doesn't matter, only null/non-null).
            toolIndex = 0;

            var argumentsRawExisting = ExtractField(ExtractField(toolCall, "function"), "arguments");
            if (argumentsRawExisting is not null)
            {
                var argsJsonExisting = NormalizeArguments(argumentsRawExisting);
                if (toolAggregates.TryGetValue(existingAnthropicIndex, out var aggregate))
                {
                    aggregate.Arguments.Append(argsJsonExisting);
                }
                createdEvents.Add(EmitContentBlockDelta(existingAnthropicIndex, new
                {
                    type = "input_json_delta",
                    partial_json = argsJsonExisting
                }));
            }

            return (toolIndex, lastToolIndex, createdEvents);
        }

        toolIndex = 0;
        lastToolIndex += 1;
        var anthropicToolIndex = lastToolIndex;
        toolKeyToAnthropicIndex[toolKey] = anthropicToolIndex;

        var function = ExtractField(toolCall, "function");
        var name = ExtractString(function, "name") ?? string.Empty;
        var stableToolId = ExtractString(toolCall, "id") ?? $"toolu_{Guid.NewGuid():N}";

        toolAggregates[anthropicToolIndex] = new ToolUseAggregate
        {
            Id = stableToolId,
            Name = name
        };

        createdEvents.Add(EmitContentBlockStart(anthropicToolIndex, new
        {
            type = "tool_use",
            id = stableToolId,
            name,
            input = new Dictionary<string, object?>()
        }));

        var argumentsRaw = ExtractField(ExtractField(toolCall, "function"), "arguments");
        if (argumentsRaw is not null)
        {
            var argsJson = NormalizeArguments(argumentsRaw);
            toolAggregates[anthropicToolIndex].Arguments.Append(argsJson);
            createdEvents.Add(EmitContentBlockDelta(anthropicToolIndex, new
            {
                type = "input_json_delta",
                partial_json = argsJson
            }));
        }

        return (toolIndex, lastToolIndex, createdEvents);
    }

    private static object ParseToolInputJson(string? rawJson)
    {
        if (string.IsNullOrWhiteSpace(rawJson))
        {
            return new Dictionary<string, object?>();
        }

        try
        {
            return JsonSerializer.Deserialize<object>(rawJson) ?? new Dictionary<string, object?>();
        }
        catch (JsonException)
        {
            return new Dictionary<string, object?> { ["raw"] = rawJson };
        }
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
