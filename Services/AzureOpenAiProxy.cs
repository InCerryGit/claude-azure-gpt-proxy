using System.Diagnostics;
using System.Text.Encodings.Web;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Linq;
using System.Security.Cryptography;
using AzureGptProxy.Models;
using AzureGptProxy.Infrastructure;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenAI.Chat;
using OpenAI;

namespace AzureGptProxy.Services;

public sealed class AzureOpenAiProxy
{
    private readonly AzureOpenAiClientFactory _clientFactory;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<AzureOpenAiProxy> _logger;
    private readonly NormalizedAzureOpenAiOptions _azureOptions;
    private readonly AzureOpenAiOptions _rawOptions;
    private readonly ResponseLog _responseLog;

    public AzureOpenAiProxy(
        AzureOpenAiClientFactory clientFactory,
        IHttpClientFactory httpClientFactory,
        ILogger<AzureOpenAiProxy> logger,
        NormalizedAzureOpenAiOptions azureOptions,
        IOptions<AzureOpenAiOptions> rawOptions,
        ResponseLog responseLog)
    {
        _clientFactory = clientFactory;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
        _azureOptions = azureOptions;
        _rawOptions = rawOptions.Value;
        _responseLog = responseLog;
    }

    public async Task<object> SendAsync(MessagesRequest request, CancellationToken cancellationToken)
    {
        _responseLog.LogRequest(request, isStream: false);
        var stopwatch = Stopwatch.StartNew();
        var payload = AnthropicConversion.ConvertAnthropicToAzure(request, _logger, _azureOptions);
        request.ResolvedAzureModel ??= payload["model"]?.ToString();
        var isResponses = IsResponsesModel(payload) || HasMultimodalContent(payload);
        if (!isResponses)
        {
            NormalizeOpenAiMessages(payload);
        }
        _logger.LogInformation(
            "Azure request start model={Model} responses={IsResponses} messages={MessageCount} tools={ToolCount}",
            request.ResolvedAzureModel ?? request.Model,
            isResponses,
            request.Messages.Count,
            request.Tools?.Count ?? 0);

        using var scope = _logger.BeginScope(new Dictionary<string, object?>
        {
            ["azureModel"] = request.ResolvedAzureModel ?? request.Model,
            ["azureResponses"] = isResponses
        });

        if (isResponses)
        {
            var responsePayload = await SendResponsesAsync(payload, cancellationToken);
            _responseLog.LogAzureResponse(responsePayload);
            _logger.LogInformation("Azure responses completed elapsedMs={ElapsedMs}", stopwatch.ElapsedMilliseconds);
            return responsePayload;
        }

        var client = _clientFactory.CreateClient();
        var deployment = ExtractDeployment(payload);

        var messages = BuildChatMessages(payload);
        var options = BuildChatOptions(payload);

        _logger.LogInformation(
            "Azure chat request deployment={Deployment} maxTokens={MaxTokens} temperature={Temperature} topP={TopP} tools={ToolCount}",
            deployment,
            options.MaxOutputTokenCount,
            options.Temperature,
            options.TopP,
            options.Tools.Count);

        var chatClient = client.GetChatClient(deployment);
        var response = await chatClient.CompleteChatAsync(messages, options, cancellationToken);
        _logger.LogInformation(
            "Azure chat completed deployment={Deployment} finishReason={FinishReason} inputTokens={InputTokens} outputTokens={OutputTokens} elapsedMs={ElapsedMs}",
            deployment,
            response.Value.FinishReason.ToString().ToLowerInvariant(),
            response.Value.Usage.InputTokenCount,
            response.Value.Usage.OutputTokenCount,
            stopwatch.ElapsedMilliseconds);

        var converted = ConvertChatResponse(response.Value);
        _responseLog.LogAzureResponse(converted);
        return converted;
    }

    public async IAsyncEnumerable<Dictionary<string, object?>> StreamAsync(
        MessagesRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        _responseLog.LogRequest(request, isStream: true);
        var stopwatch = Stopwatch.StartNew();
        var payload = AnthropicConversion.ConvertAnthropicToAzure(request, _logger, _azureOptions);
        request.ResolvedAzureModel ??= payload["model"]?.ToString();
        var isResponses = IsResponsesModel(payload) || HasMultimodalContent(payload);
        if (!isResponses)
        {
            NormalizeOpenAiMessages(payload);
        }
        _logger.LogInformation(
            "Azure stream start model={Model} responses={IsResponses} messages={MessageCount} tools={ToolCount}",
            request.ResolvedAzureModel ?? request.Model,
            isResponses,
            request.Messages.Count,
            request.Tools?.Count ?? 0);

        using var scope = _logger.BeginScope(new Dictionary<string, object?>
        {
            ["azureModel"] = request.ResolvedAzureModel ?? request.Model,
            ["azureResponses"] = isResponses
        });

        if (isResponses)
        {
            var responseChunkIndex = 0;
            await foreach (var chunk in StreamResponsesAsync(payload, cancellationToken))
            {
                _responseLog.LogAzureStreamChunk(responseChunkIndex, chunk);
                responseChunkIndex++;
                yield return chunk;
            }

            _logger.LogInformation("Azure responses stream completed elapsedMs={ElapsedMs}", stopwatch.ElapsedMilliseconds);
            yield break;
        }

        var client = _clientFactory.CreateClient();
        var deployment = ExtractDeployment(payload);

        var messages = BuildChatMessages(payload);
        var options = BuildChatOptions(payload);

        _logger.LogInformation(
            "Azure chat stream request deployment={Deployment} maxTokens={MaxTokens} temperature={Temperature} topP={TopP} tools={ToolCount}",
            deployment,
            options.MaxOutputTokenCount,
            options.Temperature,
            options.TopP,
            options.Tools.Count);

        var chatClient = client.GetChatClient(deployment);
        var chunkIndex = 0;
        await foreach (var update in chatClient.CompleteChatStreamingAsync(messages, options, cancellationToken))
        {
            var delta = new Dictionary<string, object?>
            {
                ["content"] = string.Concat(update.ContentUpdate.Select(part => part.Text))
            };

            if (update.ToolCallUpdates is { Count: > 0 })
            {
                delta["tool_calls"] = update.ToolCallUpdates;
            }

            var chunk = new Dictionary<string, object?>
            {
                ["choices"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["delta"] = delta,
                        ["finish_reason"] = update.FinishReason?.ToString().ToLowerInvariant()
                    }
                }
            };

            if (update.Usage is not null)
            {
                chunk["usage"] = new Dictionary<string, object?>
                {
                    ["prompt_tokens"] = update.Usage.InputTokenCount,
                    ["completion_tokens"] = update.Usage.OutputTokenCount
                };
            }

            _responseLog.LogAzureStreamChunk(chunkIndex, chunk);
            chunkIndex++;
            yield return chunk;
        }

        _logger.LogInformation("Azure chat stream completed deployment={Deployment} elapsedMs={ElapsedMs}", deployment, stopwatch.ElapsedMilliseconds);
    }

    private async Task<JsonElement> SendResponsesAsync(
        Dictionary<string, object?> payload,
        CancellationToken cancellationToken)
    {
        var endpoint = _azureOptions.ResponsesEndpoint ?? _azureOptions.Endpoint;
        if (string.IsNullOrWhiteSpace(endpoint))
        {
            throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is required for responses.");
        }

        if (string.IsNullOrWhiteSpace(_rawOptions.ApiKey))
        {
            throw new InvalidOperationException("AZURE_OPENAI_API_KEY is required for responses.");
        }

        var requestPayload = BuildResponsesRequestPayload(payload, stream: false);

        LogResponsesRequest(requestPayload, endpoint);

        using var httpClient = _httpClientFactory.CreateClient();
        using var requestMessage = new HttpRequestMessage(HttpMethod.Post, endpoint);
        requestMessage.Headers.TryAddWithoutValidation("api-key", _rawOptions.ApiKey);
        requestMessage.Content = new StringContent(
            JsonSerializer.Serialize(requestPayload),
            Encoding.UTF8,
            "application/json");

        var stopwatch = Stopwatch.StartNew();
        using var response = await httpClient.SendAsync(requestMessage, cancellationToken);
        var responseText = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            _logger.LogError(
                "Azure responses request failed status={StatusCode} reason={Reason} bodyLength={BodyLength} elapsedMs={ElapsedMs}",
                (int)response.StatusCode,
                response.ReasonPhrase ?? string.Empty,
                responseText.Length,
                stopwatch.ElapsedMilliseconds);
            throw new InvalidOperationException(
                $"Azure OpenAI responses request failed: {(int)response.StatusCode} {response.ReasonPhrase}. {responseText}");
        }

        _logger.LogInformation(
            "Azure responses request completed status={StatusCode} bodyLength={BodyLength} elapsedMs={ElapsedMs}",
            (int)response.StatusCode,
            responseText.Length,
            stopwatch.ElapsedMilliseconds);

        using var doc = JsonDocument.Parse(responseText);
        return doc.RootElement.Clone();
    }

    private async IAsyncEnumerable<Dictionary<string, object?>> StreamResponsesAsync(
        Dictionary<string, object?> payload,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var endpoint = _azureOptions.ResponsesEndpoint ?? _azureOptions.Endpoint;
        if (string.IsNullOrWhiteSpace(endpoint))
        {
            throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is required for responses.");
        }

        if (string.IsNullOrWhiteSpace(_rawOptions.ApiKey))
        {
            throw new InvalidOperationException("AZURE_OPENAI_API_KEY is required for responses.");
        }

        var requestPayload = BuildResponsesRequestPayload(payload, stream: true);
        LogResponsesRequest(requestPayload, endpoint);

        using var httpClient = _httpClientFactory.CreateClient();
        using var requestMessage = new HttpRequestMessage(HttpMethod.Post, endpoint);
        requestMessage.Headers.TryAddWithoutValidation("api-key", _rawOptions.ApiKey);
        requestMessage.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        requestMessage.Content = new StringContent(
            JsonSerializer.Serialize(requestPayload),
            Encoding.UTF8,
            "application/json");

        using var response = await httpClient.SendAsync(
            requestMessage,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new InvalidOperationException(
                $"Azure Responses streaming request failed: {(int)response.StatusCode} {response.ReasonPhrase}. {body}");
        }

        var toolIndexMap = new Dictionary<string, int>(StringComparer.Ordinal);
        var toolNameMap = new Dictionary<string, string>(StringComparer.Ordinal);
        var toolItemIdToCallId = new Dictionary<string, string>(StringComparer.Ordinal);
        var nextToolIndex = 0;
        var decoder = new SseDecoder();
        var emittedFinish = false;

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (!cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (line is null)
            {
                break;
            }

            foreach (var data in decoder.PushLine(line))
            {
                if (string.Equals(data.Trim(), "[DONE]", StringComparison.OrdinalIgnoreCase))
                {
                    emittedFinish = true;
                    yield break;
                }

                if (!TryParseJson(data, out var root))
                {
                    continue;
                }

                var type = root.TryGetProperty("type", out var typeProp) ? typeProp.GetString() : null;
                if (string.IsNullOrWhiteSpace(type))
                {
                    continue;
                }

                if (string.Equals(type, "response.output_item.added", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryBuildToolCallStart(root, toolIndexMap, toolNameMap, toolItemIdToCallId, ref nextToolIndex, out var toolCalls))
                    {
                        yield return BuildStreamingChunk(content: null, toolCalls: toolCalls, finishReason: null, usage: null);
                    }
                    continue;
                }

                if (string.Equals(type, "response.output_text.delta", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(type, "response.reasoning.summary_text.delta", StringComparison.OrdinalIgnoreCase))
                {
                    var delta = root.TryGetProperty("delta", out var deltaProp)
                        ? deltaProp.GetString() ?? string.Empty
                        : string.Empty;
                    if (!string.IsNullOrEmpty(delta))
                    {
                        yield return BuildStreamingChunk(content: delta, toolCalls: null, finishReason: null, usage: null);
                    }
                    continue;
                }

                if (string.Equals(type, "response.function_call.arguments.delta", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryBuildToolCallDelta(root, toolIndexMap, toolNameMap, toolItemIdToCallId, ref nextToolIndex, isOutputItemDelta: false, out var toolCalls))
                    {
                        yield return BuildStreamingChunk(content: null, toolCalls: toolCalls, finishReason: null, usage: null);
                    }
                    continue;
                }

                if (string.Equals(type, "response.output_item.delta", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryBuildToolCallDelta(root, toolIndexMap, toolNameMap, toolItemIdToCallId, ref nextToolIndex, isOutputItemDelta: true, out var toolCalls))
                    {
                        yield return BuildStreamingChunk(content: null, toolCalls: toolCalls, finishReason: null, usage: null);
                    }
                    continue;
                }

                if (string.Equals(type, "response.completed", StringComparison.OrdinalIgnoreCase))
                {
                    var completedToolCalls = ExtractCompletedToolCalls(root, toolIndexMap, toolNameMap, toolItemIdToCallId, ref nextToolIndex);
                    if (completedToolCalls.Count > 0)
                    {
                        yield return BuildStreamingChunk(content: null, toolCalls: completedToolCalls, finishReason: null, usage: null);
                    }
                    var finishReason = ResolveResponsesFinishReason(root);
                    var usage = ExtractResponsesUsage(root);
                    yield return BuildStreamingChunk(content: null, toolCalls: null, finishReason: finishReason, usage: usage);
                    emittedFinish = true;
                    yield break;
                }
            }
        }

        if (!emittedFinish)
        {
            yield return BuildStreamingChunk(content: null, toolCalls: null, finishReason: "stop", usage: null);
        }
    }


    private Dictionary<string, object?> BuildResponsesRequestPayload(
        Dictionary<string, object?> payload,
        bool stream)
    {
        var requestPayload = new Dictionary<string, object?>(payload)
        {
            ["stream"] = stream
        };

        if (requestPayload.TryGetValue("messages", out var messagesObj))
        {
            requestPayload.Remove("messages");
            requestPayload["input"] = NormalizeResponsesInput(messagesObj);
            if (!requestPayload.ContainsKey("instructions"))
            {
                var instructions = ExtractSystemInstructions(messagesObj);
                if (!string.IsNullOrWhiteSpace(instructions))
                {
                    requestPayload["instructions"] = instructions;
                }
            }
        }
        else if (requestPayload.TryGetValue("input", out var inputObj))
        {
            requestPayload["input"] = NormalizeResponsesInput(inputObj);
        }

        if (requestPayload.TryGetValue("max_completion_tokens", out var maxCompletionTokens))
        {
            requestPayload.Remove("max_completion_tokens");
            requestPayload["max_output_tokens"] = maxCompletionTokens;
        }

        if (requestPayload.TryGetValue("max_tokens", out var maxTokens))
        {
            requestPayload.Remove("max_tokens");
            requestPayload["max_output_tokens"] = maxTokens;
        }

        // Prefer metadata.user_id for prompt caching.
        // Claude Code forwards a stable per-session identifier via metadata.user_id.
        // This proxy uses it as Azure Responses prompt_cache_key for request-level caching.
        if (TryGetPromptCacheKeyFromMetadata(requestPayload, out var promptCacheKey) &&
            !string.IsNullOrWhiteSpace(promptCacheKey))
        {
            requestPayload["prompt_cache_key"] = promptCacheKey;
        }

        var legacyTokenKeys = requestPayload.Keys
            .Where(key => string.Equals(key, "max_completion_tokens", StringComparison.OrdinalIgnoreCase))
            .ToList();
        foreach (var key in legacyTokenKeys)
        {
            requestPayload.Remove(key);
        }

        if (requestPayload.TryGetValue("model", out var modelObj) && modelObj is string modelText)
        {
            requestPayload["model"] = NormalizeResponsesModel(modelText);
        }

        NormalizeResponsesTools(requestPayload);
        NormalizeResponsesToolChoice(requestPayload);
        if (requestPayload.ContainsKey("top_k"))
        {
            requestPayload.Remove("top_k");
        }
        return requestPayload;
    }

    private static bool TryGetPromptCacheKeyFromMetadata(
        IDictionary<string, object?> requestPayload,
        out string? promptCacheKey)
    {
        promptCacheKey = null;

        if (!requestPayload.TryGetValue("metadata", out var metadataObj) || metadataObj is null)
        {
            return false;
        }

        // metadata may be Dictionary<string, object?> or JsonElement depending on conversion path.
        if (metadataObj is JsonElement element)
        {
            if (element.ValueKind != JsonValueKind.Object)
            {
                return false;
            }

            if (!element.TryGetProperty("user_id", out var userIdProp))
            {
                return false;
            }

            var raw = userIdProp.ValueKind == JsonValueKind.String
                ? userIdProp.GetString()
                : userIdProp.GetRawText();
            promptCacheKey = NormalizePromptCacheKey(raw);
            return !string.IsNullOrWhiteSpace(promptCacheKey);
        }

        if (metadataObj is IDictionary<string, object?> dict)
        {
            if (!dict.TryGetValue("user_id", out var userIdObj) || userIdObj is null)
            {
                return false;
            }

            var raw = userIdObj switch
            {
                JsonElement userIdEl when userIdEl.ValueKind == JsonValueKind.String => userIdEl.GetString(),
                _ => userIdObj.ToString()
            };

            promptCacheKey = NormalizePromptCacheKey(raw);
            return !string.IsNullOrWhiteSpace(promptCacheKey);
        }

        return false;
    }

    private static string? NormalizePromptCacheKey(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        // Azure Responses enforces prompt_cache_key max length (currently 64).
        // We keep stable mapping by hashing long identifiers (e.g. Claude Code user/session ids).
        if (raw.Length <= 64)
        {
            return raw;
        }

        var bytes = Encoding.UTF8.GetBytes(raw);
        var hash = SHA256.HashData(bytes);
        // 32 bytes => 64 hex chars.
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static void NormalizeResponsesToolChoice(Dictionary<string, object?> requestPayload)
    {
        if (!requestPayload.TryGetValue("tool_choice", out var toolChoiceObj) || toolChoiceObj is null)
        {
            return;
        }

        object? normalized = toolChoiceObj switch
        {
            string choice => NormalizeToolChoiceString(choice),
            JsonElement element => NormalizeToolChoiceElement(element),
            IDictionary<string, object?> dict => NormalizeToolChoiceDictionary(dict),
            _ => toolChoiceObj
        };

        if (normalized is null)
        {
            requestPayload.Remove("tool_choice");
            return;
        }

        requestPayload["tool_choice"] = normalized;
    }

    private static object? NormalizeToolChoiceString(string choice)
    {
        return choice switch
        {
            "auto" => "auto",
            "none" => "none",
            "any" => "required",
            "required" => "required",
            _ => null
        };
    }

    private static object? NormalizeToolChoiceElement(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.String)
        {
            return NormalizeToolChoiceString(element.GetString() ?? string.Empty);
        }

        if (element.ValueKind == JsonValueKind.Object)
        {
            var dict = ConvertJsonObjectToDictionary(element);
            return NormalizeToolChoiceDictionary(dict);
        }

        return null;
    }

    private static object? NormalizeToolChoiceDictionary(IDictionary<string, object?> dict)
    {
        var type = dict.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
        var name = dict.TryGetValue("name", out var nameObj) ? nameObj?.ToString() : null;

        if (string.Equals(type, "tool", StringComparison.OrdinalIgnoreCase))
        {
            return string.IsNullOrWhiteSpace(name)
                ? null
                : new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["name"] = name
                };
        }

        if (string.Equals(type, "function", StringComparison.OrdinalIgnoreCase))
        {
            return string.IsNullOrWhiteSpace(name)
                ? null
                : new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["name"] = name
                };
        }

        if (string.Equals(type, "any", StringComparison.OrdinalIgnoreCase))
        {
            return "required";
        }

        if (string.Equals(type, "auto", StringComparison.OrdinalIgnoreCase))
        {
            return "auto";
        }

        if (string.Equals(type, "none", StringComparison.OrdinalIgnoreCase))
        {
            return "none";
        }

        if (string.Equals(type, "required", StringComparison.OrdinalIgnoreCase))
        {
            return "required";
        }

        return null;
    }

    private static bool TryParseJson(string data, out JsonElement root)
    {
        root = default;
        try
        {
            var doc = JsonDocument.Parse(data);
            root = doc.RootElement.Clone();
            doc.Dispose();
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static bool TryBuildToolCallStart(
        JsonElement root,
        Dictionary<string, int> toolIndexMap,
        Dictionary<string, string> toolNameMap,
        Dictionary<string, string> toolItemIdToCallId,
        ref int nextToolIndex,
        out List<Dictionary<string, object?>> toolCalls)
    {
        toolCalls = new List<Dictionary<string, object?>>();

        if (!root.TryGetProperty("item", out var item) || item.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        var itemType = item.TryGetProperty("type", out var itemTypeProp) ? itemTypeProp.GetString() : null;
        if (!string.Equals(itemType, "function_call", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var callId = item.TryGetProperty("call_id", out var callIdProp) ? callIdProp.GetString() : null;
        var itemId = item.TryGetProperty("id", out var itemIdProp) ? itemIdProp.GetString() : null;
        var name = item.TryGetProperty("name", out var nameProp) ? nameProp.GetString() : null;

        if (string.IsNullOrWhiteSpace(callId) && !string.IsNullOrWhiteSpace(itemId))
        {
            callId = itemId;
        }

        if (!string.IsNullOrWhiteSpace(itemId) && !string.IsNullOrWhiteSpace(callId))
        {
            toolItemIdToCallId[itemId] = callId;
        }

        if (string.IsNullOrWhiteSpace(callId) || string.IsNullOrWhiteSpace(name))
        {
            return false;
        }

        if (!toolIndexMap.TryGetValue(callId, out var index))
        {
            index = nextToolIndex++;
            toolIndexMap[callId] = index;
        }

        toolNameMap[callId] = name;
        var arguments = string.Empty;
        if (item.TryGetProperty("arguments", out var argumentsProp))
        {
            arguments = argumentsProp.ValueKind == JsonValueKind.String
                ? argumentsProp.GetString() ?? string.Empty
                : argumentsProp.GetRawText();
        }
        else if (item.TryGetProperty("arguments_delta", out var argumentsDeltaProp))
        {
            arguments = argumentsDeltaProp.ValueKind == JsonValueKind.String
                ? argumentsDeltaProp.GetString() ?? string.Empty
                : argumentsDeltaProp.GetRawText();
        }
        else if (item.TryGetProperty("input", out var inputProp))
        {
            arguments = inputProp.ValueKind == JsonValueKind.String
                ? inputProp.GetString() ?? string.Empty
                : inputProp.GetRawText();
        }
        else if (item.TryGetProperty("input_json", out var inputJsonProp))
        {
            arguments = inputJsonProp.ValueKind == JsonValueKind.String
                ? inputJsonProp.GetString() ?? string.Empty
                : inputJsonProp.GetRawText();
        }
        else if (item.TryGetProperty("input_json_delta", out var inputJsonDeltaProp))
        {
            arguments = inputJsonDeltaProp.ValueKind == JsonValueKind.String
                ? inputJsonDeltaProp.GetString() ?? string.Empty
                : inputJsonDeltaProp.GetRawText();
        }

        if (string.IsNullOrWhiteSpace(arguments) &&
            item.TryGetProperty("function", out var functionProp) &&
            functionProp.ValueKind == JsonValueKind.Object)
        {
            // Responses 事件可能把参数挂在 item.function 下。
            if (functionProp.TryGetProperty("arguments", out var functionArgumentsProp))
            {
                arguments = functionArgumentsProp.ValueKind == JsonValueKind.String
                    ? functionArgumentsProp.GetString() ?? string.Empty
                    : functionArgumentsProp.GetRawText();
            }
            else if (functionProp.TryGetProperty("arguments_delta", out var functionArgumentsDeltaProp))
            {
                arguments = functionArgumentsDeltaProp.ValueKind == JsonValueKind.String
                    ? functionArgumentsDeltaProp.GetString() ?? string.Empty
                    : functionArgumentsDeltaProp.GetRawText();
            }
            else if (functionProp.TryGetProperty("input", out var functionInputProp))
            {
                arguments = functionInputProp.ValueKind == JsonValueKind.String
                    ? functionInputProp.GetString() ?? string.Empty
                    : functionInputProp.GetRawText();
            }
            else if (functionProp.TryGetProperty("input_json", out var functionInputJsonProp))
            {
                arguments = functionInputJsonProp.ValueKind == JsonValueKind.String
                    ? functionInputJsonProp.GetString() ?? string.Empty
                    : functionInputJsonProp.GetRawText();
            }
            else if (functionProp.TryGetProperty("input_json_delta", out var functionInputJsonDeltaProp))
            {
                arguments = functionInputJsonDeltaProp.ValueKind == JsonValueKind.String
                    ? functionInputJsonDeltaProp.GetString() ?? string.Empty
                    : functionInputJsonDeltaProp.GetRawText();
            }
        }

        toolCalls.Add(new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = callId,
            ["type"] = "function",
            ["function"] = new Dictionary<string, object?>
            {
                ["name"] = name,
                ["arguments"] = arguments
            }
        });

        return true;
    }

    private static bool TryBuildToolCallDelta(
        JsonElement root,
        Dictionary<string, int> toolIndexMap,
        Dictionary<string, string> toolNameMap,
        Dictionary<string, string> toolItemIdToCallId,
        ref int nextToolIndex,
        bool isOutputItemDelta,
        out List<Dictionary<string, object?>> toolCalls)
    {
        toolCalls = new List<Dictionary<string, object?>>();

        string? callId = root.TryGetProperty("call_id", out var callIdProp) ? callIdProp.GetString() : null;
        if (string.IsNullOrWhiteSpace(callId) && root.TryGetProperty("item", out var item) && item.ValueKind == JsonValueKind.Object)
        {
            callId = item.TryGetProperty("call_id", out var itemCallIdProp) ? itemCallIdProp.GetString() : null;
        }

        string? itemId = root.TryGetProperty("item_id", out var itemIdProp) ? itemIdProp.GetString() : null;
        if (string.IsNullOrWhiteSpace(itemId))
        {
            itemId = root.TryGetProperty("id", out var rootIdProp) ? rootIdProp.GetString() : null;
        }

        if (string.IsNullOrWhiteSpace(itemId) && root.TryGetProperty("item", out var rootItem) && rootItem.ValueKind == JsonValueKind.Object)
        {
            itemId = rootItem.TryGetProperty("id", out var itemIdProp2) ? itemIdProp2.GetString() : null;
        }

        if (string.IsNullOrWhiteSpace(callId) && !string.IsNullOrWhiteSpace(itemId) && toolItemIdToCallId.TryGetValue(itemId, out var mappedCallId))
        {
            callId = mappedCallId;
        }

        if (string.IsNullOrWhiteSpace(callId) && !string.IsNullOrWhiteSpace(itemId))
        {
            callId = itemId;
        }

        string delta = string.Empty;
        if (!isOutputItemDelta && root.TryGetProperty("delta", out var deltaProp))
        {
            delta = deltaProp.ValueKind == JsonValueKind.String ? deltaProp.GetString() ?? string.Empty : deltaProp.GetRawText();
        }
        else if (root.TryGetProperty("arguments", out var argumentsProp))
        {
            delta = argumentsProp.ValueKind == JsonValueKind.String ? argumentsProp.GetString() ?? string.Empty : argumentsProp.GetRawText();
        }
        else if (root.TryGetProperty("arguments_delta", out var argumentsDeltaProp))
        {
            delta = argumentsDeltaProp.ValueKind == JsonValueKind.String ? argumentsDeltaProp.GetString() ?? string.Empty : argumentsDeltaProp.GetRawText();
        }
        else if (!isOutputItemDelta && root.TryGetProperty("input", out var inputProp))
        {
            delta = inputProp.ValueKind == JsonValueKind.String ? inputProp.GetString() ?? string.Empty : inputProp.GetRawText();
        }
        else if (root.TryGetProperty("input_json_delta", out var inputJsonDeltaProp))
        {
            delta = inputJsonDeltaProp.ValueKind == JsonValueKind.String ? inputJsonDeltaProp.GetString() ?? string.Empty : inputJsonDeltaProp.GetRawText();
        }
        else if (root.TryGetProperty("item", out var itemProp) && itemProp.ValueKind == JsonValueKind.Object)
        {
            if (isOutputItemDelta)
            {
                var itemType = itemProp.TryGetProperty("type", out var itemTypeProp) ? itemTypeProp.GetString() : null;
                if (!string.Equals(itemType, "function_call", StringComparison.OrdinalIgnoreCase))
                {
                    return false;
                }
            }

            if (!isOutputItemDelta && itemProp.TryGetProperty("delta", out var itemDeltaProp))
            {
                delta = itemDeltaProp.ValueKind == JsonValueKind.String ? itemDeltaProp.GetString() ?? string.Empty : itemDeltaProp.GetRawText();
            }
            else if (itemProp.TryGetProperty("arguments", out var itemArgumentsProp))
            {
                delta = itemArgumentsProp.ValueKind == JsonValueKind.String ? itemArgumentsProp.GetString() ?? string.Empty : itemArgumentsProp.GetRawText();
            }
            else if (itemProp.TryGetProperty("arguments_delta", out var itemArgumentsDeltaProp))
            {
                delta = itemArgumentsDeltaProp.ValueKind == JsonValueKind.String ? itemArgumentsDeltaProp.GetString() ?? string.Empty : itemArgumentsDeltaProp.GetRawText();
            }
            else if (!isOutputItemDelta && itemProp.TryGetProperty("input", out var itemInputProp))
            {
                delta = itemInputProp.ValueKind == JsonValueKind.String ? itemInputProp.GetString() ?? string.Empty : itemInputProp.GetRawText();
            }
            else if (itemProp.TryGetProperty("input_json_delta", out var itemInputJsonDeltaProp))
            {
                delta = itemInputJsonDeltaProp.ValueKind == JsonValueKind.String ? itemInputJsonDeltaProp.GetString() ?? string.Empty : itemInputJsonDeltaProp.GetRawText();
            }
            else if (itemProp.TryGetProperty("function", out var itemFunctionProp) && itemFunctionProp.ValueKind == JsonValueKind.Object)
            {
                // Responses 事件可能把参数挂在 item.function 下。
                if (itemFunctionProp.TryGetProperty("arguments", out var itemFunctionArgumentsProp))
                {
                    delta = itemFunctionArgumentsProp.ValueKind == JsonValueKind.String ? itemFunctionArgumentsProp.GetString() ?? string.Empty : itemFunctionArgumentsProp.GetRawText();
                }
                else if (itemFunctionProp.TryGetProperty("arguments_delta", out var itemFunctionArgumentsDeltaProp))
                {
                    delta = itemFunctionArgumentsDeltaProp.ValueKind == JsonValueKind.String ? itemFunctionArgumentsDeltaProp.GetString() ?? string.Empty : itemFunctionArgumentsDeltaProp.GetRawText();
                }
                else if (!isOutputItemDelta && itemFunctionProp.TryGetProperty("input", out var itemFunctionInputProp))
                {
                    delta = itemFunctionInputProp.ValueKind == JsonValueKind.String ? itemFunctionInputProp.GetString() ?? string.Empty : itemFunctionInputProp.GetRawText();
                }
                else if (itemFunctionProp.TryGetProperty("input_json_delta", out var itemFunctionInputJsonDeltaProp))
                {
                    delta = itemFunctionInputJsonDeltaProp.ValueKind == JsonValueKind.String ? itemFunctionInputJsonDeltaProp.GetString() ?? string.Empty : itemFunctionInputJsonDeltaProp.GetRawText();
                }
            }
        }

        if (string.IsNullOrWhiteSpace(callId) || string.IsNullOrEmpty(delta))
        {
            return false;
        }

        if (!toolIndexMap.TryGetValue(callId, out var index))
        {
            index = nextToolIndex++;
            toolIndexMap[callId] = index;
        }

        toolNameMap.TryGetValue(callId, out var name);
        toolCalls.Add(new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = callId,
            ["type"] = "function",
            ["function"] = new Dictionary<string, object?>
            {
                ["name"] = name,
                ["arguments"] = delta
            }
        });

        return true;
    }

    private static List<Dictionary<string, object?>> ExtractCompletedToolCalls(
        JsonElement root,
        Dictionary<string, int> toolIndexMap,
        Dictionary<string, string> toolNameMap,
        Dictionary<string, string> toolItemIdToCallId,
        ref int nextToolIndex)
    {
        var toolCalls = new List<Dictionary<string, object?>>();

        if (!root.TryGetProperty("response", out var response) ||
            !response.TryGetProperty("output", out var output) ||
            output.ValueKind != JsonValueKind.Array)
        {
            return toolCalls;
        }

        foreach (var item in output.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            var itemType = item.TryGetProperty("type", out var typeProp) ? typeProp.GetString() : null;
            if (!string.Equals(itemType, "function_call", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var callId = item.TryGetProperty("call_id", out var callIdProp) ? callIdProp.GetString() : null;
            var itemId = item.TryGetProperty("id", out var itemIdProp) ? itemIdProp.GetString() : null;
            var name = item.TryGetProperty("name", out var nameProp) ? nameProp.GetString() : null;

            if (string.IsNullOrWhiteSpace(callId) && !string.IsNullOrWhiteSpace(itemId))
            {
                callId = itemId;
            }

            if (!string.IsNullOrWhiteSpace(itemId) && !string.IsNullOrWhiteSpace(callId))
            {
                toolItemIdToCallId[itemId] = callId;
            }

            if (item.TryGetProperty("function", out var functionProp) && functionProp.ValueKind == JsonValueKind.Object)
            {
                if (string.IsNullOrWhiteSpace(name) && functionProp.TryGetProperty("name", out var functionNameProp))
                {
                    name = functionNameProp.GetString();
                }
            }

            if (string.IsNullOrWhiteSpace(callId) || string.IsNullOrWhiteSpace(name))
            {
                continue;
            }

            if (!toolIndexMap.TryGetValue(callId, out var index))
            {
                index = nextToolIndex++;
                toolIndexMap[callId] = index;
            }

            toolNameMap[callId] = name;

            var arguments = string.Empty;
            if (item.TryGetProperty("arguments", out var argumentsProp))
            {
                arguments = argumentsProp.ValueKind == JsonValueKind.String
                    ? argumentsProp.GetString() ?? string.Empty
                    : argumentsProp.GetRawText();
            }
            else if (item.TryGetProperty("input", out var inputProp))
            {
                arguments = inputProp.ValueKind == JsonValueKind.String
                    ? inputProp.GetString() ?? string.Empty
                    : inputProp.GetRawText();
            }
            else if (item.TryGetProperty("input_json", out var inputJsonProp))
            {
                arguments = inputJsonProp.ValueKind == JsonValueKind.String
                    ? inputJsonProp.GetString() ?? string.Empty
                    : inputJsonProp.GetRawText();
            }
            else if (item.TryGetProperty("function", out var functionArgProp) && functionArgProp.ValueKind == JsonValueKind.Object)
            {
                if (functionArgProp.TryGetProperty("arguments", out var functionArgumentsProp))
                {
                    arguments = functionArgumentsProp.ValueKind == JsonValueKind.String
                        ? functionArgumentsProp.GetString() ?? string.Empty
                        : functionArgumentsProp.GetRawText();
                }
                else if (functionArgProp.TryGetProperty("input", out var functionInputProp))
                {
                    arguments = functionInputProp.ValueKind == JsonValueKind.String
                        ? functionInputProp.GetString() ?? string.Empty
                        : functionInputProp.GetRawText();
                }
                else if (functionArgProp.TryGetProperty("input_json", out var functionInputJsonProp))
                {
                    arguments = functionInputJsonProp.ValueKind == JsonValueKind.String
                        ? functionInputJsonProp.GetString() ?? string.Empty
                        : functionInputJsonProp.GetRawText();
                }
            }

            toolCalls.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = callId,
                ["type"] = "function",
                ["function"] = new Dictionary<string, object?>
                {
                    ["name"] = name,
                    ["arguments"] = arguments
                }
            });
        }

        return toolCalls;
    }

    private static string ResolveResponsesFinishReason(JsonElement root)
    {
        if (root.TryGetProperty("response", out var response) &&
            response.TryGetProperty("output", out var output) &&
            output.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in output.EnumerateArray())
            {
                var itemType = item.TryGetProperty("type", out var typeProp) ? typeProp.GetString() : null;
                if (string.Equals(itemType, "function_call", StringComparison.OrdinalIgnoreCase))
                {
                    return "tool_calls";
                }
            }
        }

        return "stop";
    }

    private static Dictionary<string, object?>? ExtractResponsesUsage(JsonElement root)
    {
        if (!root.TryGetProperty("response", out var response) ||
            !response.TryGetProperty("usage", out var usage) ||
            usage.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var promptTokens = usage.TryGetProperty("input_tokens", out var inputProp) && inputProp.TryGetInt32(out var input)
            ? input
            : usage.TryGetProperty("prompt_tokens", out var promptProp) && promptProp.TryGetInt32(out var prompt)
                ? prompt
                : 0;

        var completionTokens = usage.TryGetProperty("output_tokens", out var outputProp) && outputProp.TryGetInt32(out var output)
            ? output
            : usage.TryGetProperty("completion_tokens", out var completionProp) && completionProp.TryGetInt32(out var completion)
                ? completion
                : 0;

        return new Dictionary<string, object?>
        {
            ["prompt_tokens"] = promptTokens,
            ["completion_tokens"] = completionTokens
        };
    }

    private static Dictionary<string, object?> BuildStreamingChunk(
        string? content,
        object? toolCalls,
        string? finishReason,
        Dictionary<string, object?>? usage)
    {
        var delta = new Dictionary<string, object?>();
        if (content is not null)
        {
            delta["content"] = content;
        }

        if (toolCalls is not null)
        {
            delta["tool_calls"] = toolCalls;
        }

        var chunk = new Dictionary<string, object?>
        {
            ["choices"] = new[]
            {
                new Dictionary<string, object?>
                {
                    ["delta"] = delta,
                    ["finish_reason"] = finishReason
                }
            }
        };

        if (usage is not null)
        {
            chunk["usage"] = usage;
        }

        return chunk;
    }

    private void LogResponsesRequest(Dictionary<string, object?> requestPayload, string endpoint)
    {
        if (requestPayload.TryGetValue("tools", out var toolsPayload) && toolsPayload is IEnumerable<object> toolsList)
        {
            _logger.LogInformation("Azure responses tools count: {Count}", toolsList.Count());
        }

        if (requestPayload.TryGetValue("input", out var inputPayload) && inputPayload is IEnumerable<object> inputList)
        {
            _logger.LogInformation("Azure responses input items: {Count}", inputList.Count());
        }

        _logger.LogInformation(
            "Azure responses request: endpoint={Endpoint}, model={Model}",
            endpoint,
            requestPayload.TryGetValue("model", out var resolvedModel) ? resolvedModel : "(missing)");

        if (_logger.IsEnabled(LogLevel.Debug))
        {
            var jsonOptions = new JsonSerializerOptions
            {
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
            };
            var requestJson = JsonSerializer.Serialize(requestPayload, jsonOptions);
            _logger.LogDebug("Azure responses request payload: {RequestPayload}", requestJson);
        }
    }


    private static string NormalizeResponsesModel(string model)
    {
        if (model.StartsWith("azure/responses/", StringComparison.OrdinalIgnoreCase))
        {
            return model["azure/responses/".Length..];
        }

        if (model.StartsWith("azure/", StringComparison.OrdinalIgnoreCase))
        {
            return model["azure/".Length..];
        }

        return model;
    }

    private static void NormalizeResponsesTools(Dictionary<string, object?> requestPayload)
    {
        if (!requestPayload.TryGetValue("tools", out var toolsObj) || toolsObj is null)
        {
            return;
        }

        IEnumerable<object> tools = toolsObj switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
            IEnumerable<object> list => list,
            _ => Array.Empty<object>()
        };

        var normalizedTools = new List<Dictionary<string, object?>>();
        foreach (var tool in tools)
        {
            Dictionary<string, object?> toolDict = tool switch
            {
                JsonElement element when element.ValueKind == JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
                IDictionary<string, object?> dict => new Dictionary<string, object?>(dict),
                _ => new Dictionary<string, object?>()
            };

            if (toolDict.TryGetValue("function", out var functionObj))
            {
                var functionDict = functionObj switch
                {
                    JsonElement functionElement when functionElement.ValueKind == JsonValueKind.Object =>
                        ConvertJsonObjectToDictionary(functionElement),
                    IDictionary<string, object?> dict => new Dictionary<string, object?>(dict),
                    _ => new Dictionary<string, object?>()
                };

                toolDict.Remove("function");
                if (functionDict.TryGetValue("name", out var name))
                {
                    toolDict["name"] = name is JsonElement nameElement && nameElement.ValueKind == JsonValueKind.String
                        ? nameElement.GetString()
                        : name?.ToString();
                }

                if (functionDict.TryGetValue("description", out var description))
                {
                    toolDict["description"] = description;
                }

                if (functionDict.TryGetValue("parameters", out var parameters))
                {
                    toolDict["parameters"] = parameters;
                }
            }

            if (!toolDict.TryGetValue("name", out var toolName) || string.IsNullOrWhiteSpace(toolName?.ToString()))
            {
                throw new InvalidOperationException("Tool name is required.");
            }

            normalizedTools.Add(toolDict);
        }

        requestPayload["tools"] = normalizedTools;
    }

    private static object? NormalizeResponsesInput(object? input)
    {
        if (input is null)
        {
            return null;
        }

        var items = new List<Dictionary<string, object?>>();
        switch (input)
        {
            case JsonElement element when element.ValueKind == JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                {
                    if (item.ValueKind == JsonValueKind.Object)
                    {
                        items.Add(ConvertJsonObjectToDictionary(item));
                    }
                }
                break;
            case IEnumerable<object> list:
                foreach (var item in list)
                {
                    if (item is JsonElement itemElement && itemElement.ValueKind == JsonValueKind.Object)
                    {
                        items.Add(ConvertJsonObjectToDictionary(itemElement));
                    }
                    else if (item is IDictionary<string, object?> dict)
                    {
                        items.Add(new Dictionary<string, object?>(dict));
                    }
                }
                break;
            default:
                return input;
        }

        var normalized = new List<object>();
        string? lastFunctionCallId = null;

        for (var idx = 0; idx < items.Count; idx++)
        {
            var message = items[idx];
            if (!message.TryGetValue("role", out var roleObj) || roleObj is null)
            {
                continue;
            }

            var role = roleObj.ToString() ?? "user";

            if (string.Equals(role, "system", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase))
            {
                if (message.TryGetValue("tool_calls", out var toolCallsObj) && toolCallsObj is not null)
                {
                    foreach (var toolCall in EnumerateToolCalls(toolCallsObj))
                    {
                        if (TryConvertToolCallToFunctionCall(toolCall, items, idx, out var functionCall))
                        {
                            if (functionCall.TryGetValue("call_id", out var callIdObj))
                            {
                                lastFunctionCallId = callIdObj?.ToString() ?? lastFunctionCallId;
                            }
                            normalized.Add(functionCall);
                        }
                    }
                }
            }

            if (string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase))
            {
                if (TryConvertToolRoleToFunctionOutput(message, lastFunctionCallId, out var toolOutput))
                {
                    if (toolOutput.TryGetValue("call_id", out var toolCallIdObj))
                    {
                        lastFunctionCallId = toolCallIdObj?.ToString() ?? lastFunctionCallId;
                    }
                    normalized.Add(toolOutput);
                }
                continue;
            }

            var normalizedMessage = new Dictionary<string, object?>
            {
                ["type"] = "message",
                ["role"] = role,
                ["content"] = NormalizeResponsesContent(role, message.TryGetValue("content", out var contentObj) ? contentObj : null)
            };

            normalized.Add(normalizedMessage);
        }

        return normalized;
    }

    private static string? ExtractSystemInstructions(object? messagesObj)
    {
        IEnumerable<object> messages = messagesObj switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
            IEnumerable<object> list => list,
            _ => Array.Empty<object>()
        };

        var buffer = new List<string>();
        foreach (var message in messages)
        {
            if (message is not IDictionary<string, object?> dict)
            {
                if (message is JsonElement element && element.ValueKind == JsonValueKind.Object)
                {
                    dict = ConvertJsonObjectToDictionary(element);
                }
                else
                {
                    continue;
                }
            }

            if (!dict.TryGetValue("role", out var roleObj) || !string.Equals(roleObj?.ToString(), "system", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!dict.TryGetValue("content", out var contentObj) || contentObj is null)
            {
                continue;
            }

            var serialized = NormalizeSystemContentForInstructions(contentObj);
            if (!string.IsNullOrWhiteSpace(serialized))
            {
                buffer.Add(serialized);
            }
        }

        return buffer.Count == 0 ? null : string.Join("\n\n", buffer);
    }

    private static string NormalizeSystemContentForInstructions(object contentObj)
    {
        if (contentObj is string text)
        {
            return text;
        }

        if (contentObj is JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.String => element.GetString() ?? string.Empty,
                JsonValueKind.Array => element.GetRawText(),
                JsonValueKind.Object => element.GetRawText(),
                _ => string.Empty
            };
        }

        if (contentObj is IEnumerable<object> list)
        {
            return JsonSerializer.Serialize(list);
        }

        return contentObj.ToString() ?? string.Empty;
    }

    private static IEnumerable<object> EnumerateToolCalls(object toolCallsObj)
    {
        if (toolCallsObj is JsonElement element && element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                yield return item;
            }
            yield break;
        }

        if (toolCallsObj is IEnumerable<object> list)
        {
            foreach (var item in list)
            {
                yield return item;
            }
            yield break;
        }

        yield return toolCallsObj;
    }

    private static bool TryConvertToolCallToFunctionCall(
        object toolCall,
        List<Dictionary<string, object?>> messages,
        int currentIndex,
        out Dictionary<string, object?> functionCall)
    {
        functionCall = new Dictionary<string, object?>();
        Dictionary<string, object?>? dict = toolCall switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
            IDictionary<string, object?> map => new Dictionary<string, object?>(map),
            _ => null
        };

        if (dict is null)
        {
            return false;
        }

        var callId = ExtractToolCallId(dict) ?? PeekNextToolCallId(messages, currentIndex);
        var name = ExtractToolCallName(dict);
        var arguments = ExtractToolCallArguments(dict);
        if (string.IsNullOrWhiteSpace(callId) || string.IsNullOrWhiteSpace(name))
        {
            return false;
        }

        functionCall["type"] = "function_call";
        functionCall["call_id"] = callId;
        functionCall["name"] = name;
        functionCall["arguments"] = arguments;
        return true;
    }

    private static string? ExtractToolCallId(Dictionary<string, object?> dict)
    {
        if (dict.TryGetValue("id", out var idObj) && !string.IsNullOrWhiteSpace(idObj?.ToString()))
        {
            return idObj?.ToString();
        }

        if (dict.TryGetValue("call_id", out var callIdObj) && !string.IsNullOrWhiteSpace(callIdObj?.ToString()))
        {
            return callIdObj?.ToString();
        }

        if (dict.TryGetValue("tool_call_id", out var toolCallIdObj) && !string.IsNullOrWhiteSpace(toolCallIdObj?.ToString()))
        {
            return toolCallIdObj?.ToString();
        }

        return null;
    }

    private static string? ExtractToolCallName(Dictionary<string, object?> dict)
    {
        if (dict.TryGetValue("name", out var nameObj) && !string.IsNullOrWhiteSpace(nameObj?.ToString()))
        {
            return nameObj?.ToString();
        }

        if (dict.TryGetValue("function", out var functionObj))
        {
            var functionDict = functionObj switch
            {
                JsonElement element when element.ValueKind == JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
                IDictionary<string, object?> map => new Dictionary<string, object?>(map),
                _ => null
            };
            if (functionDict is not null && functionDict.TryGetValue("name", out var nestedName))
            {
                return nestedName?.ToString();
            }
        }

        return null;
    }

    private static string ExtractToolCallArguments(Dictionary<string, object?> dict)
    {
        object? argumentsObj = null;
        if (dict.TryGetValue("arguments", out var directArgs))
        {
            argumentsObj = directArgs;
        }
        else if (dict.TryGetValue("function", out var functionObj))
        {
            if (functionObj is JsonElement element && element.ValueKind == JsonValueKind.Object && element.TryGetProperty("arguments", out var argProp))
            {
                argumentsObj = argProp;
            }
            else if (functionObj is IDictionary<string, object?> funcDict && funcDict.TryGetValue("arguments", out var nestedArgs))
            {
                argumentsObj = nestedArgs;
            }
        }

        if (argumentsObj is null)
        {
            return "{}";
        }

        if (argumentsObj is string str)
        {
            return str;
        }

        if (argumentsObj is JsonElement argElement)
        {
            return argElement.ValueKind == JsonValueKind.String
                ? argElement.GetString() ?? "{}"
                : argElement.GetRawText();
        }

        return JsonSerializer.Serialize(argumentsObj);
    }

    private static bool TryConvertToolRoleToFunctionOutput(
        Dictionary<string, object?> message,
        string? lastFunctionCallId,
        out Dictionary<string, object?> functionOutput)
    {
        functionOutput = new Dictionary<string, object?>();
        if (!message.TryGetValue("role", out var roleObj) ||
            !string.Equals(roleObj?.ToString(), "tool", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var callId = message.TryGetValue("tool_call_id", out var callIdObj)
            ? callIdObj?.ToString()
            : message.TryGetValue("call_id", out var altCallIdObj)
                ? altCallIdObj?.ToString()
                : null;

        if (string.IsNullOrWhiteSpace(callId))
        {
            throw new InvalidOperationException("Tool output is missing tool_call_id/call_id; cannot safely associate tool output with a prior tool call.");
        }
        var output = message.TryGetValue("content", out var contentObj)
            ? NormalizeToolOutput(contentObj)
            : string.Empty;

        functionOutput["type"] = "function_call_output";
        functionOutput["call_id"] = callId ?? string.Empty;
        functionOutput["output"] = output;
        return true;
    }

    private static string? PeekNextToolCallId(List<Dictionary<string, object?>> messages, int currentIndex)
    {
        for (var i = currentIndex + 1; i < messages.Count; i++)
        {
            var message = messages[i];
            if (!message.TryGetValue("role", out var roleObj))
            {
                continue;
            }

            if (!string.Equals(roleObj?.ToString(), "tool", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (message.TryGetValue("tool_call_id", out var toolCallIdObj) && !string.IsNullOrWhiteSpace(toolCallIdObj?.ToString()))
            {
                return toolCallIdObj?.ToString();
            }

            if (message.TryGetValue("call_id", out var callIdObj) && !string.IsNullOrWhiteSpace(callIdObj?.ToString()))
            {
                return callIdObj?.ToString();
            }
        }

        return null;
    }

    private static List<Dictionary<string, object?>> NormalizeResponsesContent(string role, object? content)
    {
        if (content is null)
        {
            return new List<Dictionary<string, object?>>();
        }

        if (content is string text)
        {
            return new List<Dictionary<string, object?>>
            {
                new()
                {
                    ["type"] = DefaultContentTypeForRole(role),
                    ["text"] = text
                }
            };
        }

        if (content is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String)
            {
                return new List<Dictionary<string, object?>>
                {
                    new()
                    {
                        ["type"] = DefaultContentTypeForRole(role),
                        ["text"] = element.GetString() ?? string.Empty
                    }
                };
            }

            if (element.ValueKind == JsonValueKind.Array)
            {
                var parts = new List<Dictionary<string, object?>>();
                foreach (var item in element.EnumerateArray())
                {
                    parts.Add(NormalizeContentPart(role, item));
                }
                return parts;
            }
        }

        if (content is IEnumerable<object> list)
        {
            var parts = new List<Dictionary<string, object?>>();
            foreach (var item in list)
            {
                parts.Add(NormalizeContentPart(role, item));
            }
            return parts;
        }

        return new List<Dictionary<string, object?>>
        {
            new()
            {
                ["type"] = DefaultContentTypeForRole(role),
                ["text"] = content.ToString() ?? string.Empty
            }
        };
    }

    private static Dictionary<string, object?> NormalizeContentPart(string role, object part)
    {
        if (part is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String)
            {
                return new Dictionary<string, object?>
                {
                    ["type"] = DefaultContentTypeForRole(role),
                    ["text"] = element.GetString() ?? string.Empty
                };
            }

            if (element.ValueKind == JsonValueKind.Object)
            {
                var dict = ConvertJsonObjectToDictionary(element);
                return NormalizeContentPart(role, dict);
            }
        }

        if (part is IDictionary<string, object?> partDict)
        {
            if (TryNormalizeDocumentPart(partDict, out var documentPart))
            {
                return documentPart;
            }

            if (TryNormalizeImagePart(partDict, out var imagePart))
            {
                return imagePart;
            }

            if (TryNormalizeSearchResultPart(partDict, out var searchPart))
            {
                return searchPart;
            }

            var copy = new Dictionary<string, object?>(partDict);
            var type = copy.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
            copy["type"] = NormalizeContentType(role, type) ?? DefaultContentTypeForRole(role);

            if (!RequiresTextField(copy["type"]?.ToString()))
            {
                copy.Remove("text");
                return copy;
            }

            if (!copy.ContainsKey("text") && copy.TryGetValue("content", out var contentObj))
            {
                copy["text"] = contentObj?.ToString() ?? string.Empty;
                copy.Remove("content");
            }
            if (!copy.ContainsKey("text"))
            {
                copy["text"] = string.Empty;
            }
            return copy;
        }

        return new Dictionary<string, object?>
        {
            ["type"] = DefaultContentTypeForRole(role),
            ["text"] = part.ToString() ?? string.Empty
        };
    }

    private static string DefaultContentTypeForRole(string role)
    {
        if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase))
        {
            return "output_text";
        }

        if (string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase))
        {
            return "tool_result";
        }

        return "input_text";
    }

    private static string? NormalizeContentType(string role, string? type)
    {
        if (string.IsNullOrWhiteSpace(type))
        {
            return null;
        }

        var lowered = type.ToLowerInvariant();
        if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase))
        {
            return lowered switch
            {
                "text" or "output_text" => "output_text",
                _ when lowered.StartsWith("output_") => lowered,
                _ => null
            };
        }

        if (string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase))
        {
            return lowered switch
            {
                "text" or "tool_result" or "input_text" or "output_text" => "tool_result",
                _ when lowered.StartsWith("tool_") => lowered,
                _ => null
            };
        }

        return lowered switch
        {
            "text" or "input_text" => "input_text",
            "image" or "input_image" => "input_image",
            _ when lowered.StartsWith("input_") => lowered,
            _ => null
        };
    }

    private static bool RequiresTextField(string? type)
    {
        if (string.IsNullOrWhiteSpace(type))
        {
            return true;
        }

        return type is "input_text" or "output_text" or "tool_result";
    }

    private static bool TryNormalizeDocumentPart(IDictionary<string, object?> partDict, out Dictionary<string, object?> documentPart)
    {
        documentPart = new Dictionary<string, object?>();

        if (!partDict.TryGetValue("type", out var typeObj))
        {
            return false;
        }

        var type = typeObj?.ToString();
        if (!string.Equals(type, "document", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(type, "input_document", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (!partDict.TryGetValue("source", out var sourceObj) || sourceObj is null)
        {
            return false;
        }

        if (TryBuildInputFileFromDocumentSource(sourceObj, partDict, out var inputFile))
        {
            documentPart = inputFile;
            return true;
        }

        return false;
    }

    private static bool TryBuildInputFileFromDocumentSource(
        object sourceObj,
        IDictionary<string, object?> partDict,
        out Dictionary<string, object?> inputFile)
    {
        inputFile = new Dictionary<string, object?>();

        Dictionary<string, object?>? source = sourceObj switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
            IDictionary<string, object?> map => new Dictionary<string, object?>(map),
            _ => null
        };

        if (source is null)
        {
            return false;
        }

        var sourceType = source.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
        if (string.Equals(sourceType, "base64", StringComparison.OrdinalIgnoreCase))
        {
            var data = source.TryGetValue("data", out var dataObj) ? dataObj?.ToString() : null;
            var mediaType = source.TryGetValue("media_type", out var mediaTypeObj)
                ? mediaTypeObj?.ToString()
                : "application/pdf";
            if (string.IsNullOrWhiteSpace(data))
            {
                return false;
            }

            inputFile["type"] = "input_file";
            inputFile["file_data"] = $"data:{mediaType};base64,{data}";
            inputFile["filename"] = partDict.TryGetValue("title", out var titleObj) && !string.IsNullOrWhiteSpace(titleObj?.ToString())
                ? titleObj?.ToString()
                : "document.pdf";
            return true;
        }

        if (string.Equals(sourceType, "text", StringComparison.OrdinalIgnoreCase))
        {
            var data = source.TryGetValue("data", out var dataObj) ? dataObj?.ToString() : null;
            if (string.IsNullOrWhiteSpace(data))
            {
                return false;
            }

            inputFile["type"] = "input_text";
            inputFile["text"] = data;
            return true;
        }

        if (source.TryGetValue("file_id", out var fileIdObj) && !string.IsNullOrWhiteSpace(fileIdObj?.ToString()))
        {
            inputFile["type"] = "input_file";
            inputFile["file_id"] = fileIdObj?.ToString();
            return true;
        }

        // URL/content blocks are not directly supported by Responses input_file; keep as text.
        inputFile["type"] = "input_text";
        inputFile["text"] = JsonSerializer.Serialize(partDict);
        return true;
    }

    private static bool TryNormalizeSearchResultPart(IDictionary<string, object?> partDict, out Dictionary<string, object?> searchPart)
    {
        searchPart = new Dictionary<string, object?>();

        if (!partDict.TryGetValue("type", out var typeObj))
        {
            return false;
        }

        var type = typeObj?.ToString();
        if (!string.Equals(type, "search_result", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        searchPart["type"] = "input_text";
        searchPart["text"] = JsonSerializer.Serialize(partDict);
        return true;
    }

    private static bool TryNormalizeImagePart(IDictionary<string, object?> partDict, out Dictionary<string, object?> imagePart)
    {
        imagePart = new Dictionary<string, object?>();

        if (!partDict.TryGetValue("type", out var typeObj))
        {
            return false;
        }

        var type = typeObj?.ToString();
        if (!string.Equals(type, "image", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(type, "input_image", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (partDict.TryGetValue("image_url", out var imageUrlObj))
        {
            if (TryExtractImageUrlString(imageUrlObj, out var extractedUrl))
            {
                imagePart["type"] = "input_image";
                imagePart["image_url"] = extractedUrl;
                return true;
            }
        }

        if (partDict.TryGetValue("source", out var sourceObj) && sourceObj is not null)
        {
            if (TryBuildImageUrl(sourceObj, out var imageUrl))
            {
                imagePart["type"] = "input_image";
                imagePart["image_url"] = imageUrl;
                return true;
            }
        }

        return false;
    }

    private static bool TryExtractImageUrlString(object? imageUrlObj, out string imageUrl)
    {
        imageUrl = string.Empty;

        if (imageUrlObj is null)
        {
            return false;
        }

        if (imageUrlObj is string urlString)
        {
            if (string.IsNullOrWhiteSpace(urlString))
            {
                return false;
            }

            imageUrl = urlString;
            return true;
        }

        if (imageUrlObj is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String)
            {
                var url = element.GetString();
                if (string.IsNullOrWhiteSpace(url))
                {
                    return false;
                }

                imageUrl = url;
                return true;
            }

            if (element.ValueKind == JsonValueKind.Object &&
                element.TryGetProperty("url", out var urlProp) &&
                urlProp.ValueKind == JsonValueKind.String)
            {
                var url = urlProp.GetString();
                if (string.IsNullOrWhiteSpace(url))
                {
                    return false;
                }

                imageUrl = url;
                return true;
            }

            return false;
        }

        if (imageUrlObj is IDictionary<string, object?> dict)
        {
            if (dict.TryGetValue("url", out var urlObj) && TryExtractImageUrlString(urlObj, out var nestedUrl))
            {
                imageUrl = nestedUrl;
                return true;
            }

            return false;
        }

        return false;
    }

    private static bool TryBuildImageUrl(object sourceObj, out string imageUrl)
    {
        imageUrl = string.Empty;

        Dictionary<string, object?>? dict = sourceObj switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
            IDictionary<string, object?> map => new Dictionary<string, object?>(map),
            _ => null
        };

        if (dict is null)
        {
            return false;
        }

        var type = dict.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
        if (string.Equals(type, "url", StringComparison.OrdinalIgnoreCase) &&
            dict.TryGetValue("url", out var urlObj) &&
            !string.IsNullOrWhiteSpace(urlObj?.ToString()))
        {
            imageUrl = urlObj?.ToString() ?? string.Empty;
            return true;
        }

        if (string.Equals(type, "base64", StringComparison.OrdinalIgnoreCase) &&
            dict.TryGetValue("data", out var dataObj) &&
            !string.IsNullOrWhiteSpace(dataObj?.ToString()))
        {
            var mediaType = dict.TryGetValue("media_type", out var mediaTypeObj)
                ? mediaTypeObj?.ToString()
                : "image/png";
            imageUrl = $"data:{mediaType};base64,{dataObj}";
            return true;
        }

        return false;
    }

    private static string NormalizeToolOutput(object? value)
    {
        if (value is null)
        {
            return string.Empty;
        }

        if (value is string str)
        {
            return str;
        }

        if (value is JsonElement element)
        {
            return element.ValueKind == JsonValueKind.String
                ? element.GetString() ?? string.Empty
                : element.GetRawText();
        }

        return JsonSerializer.Serialize(value);
    }

    private static Dictionary<string, object?> ConvertJsonObjectToDictionary(JsonElement element)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var prop in element.EnumerateObject())
        {
            dict[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.Object => ConvertJsonObjectToDictionary(prop.Value),
                JsonValueKind.Array => prop.Value.EnumerateArray().Select(item => item).ToList(),
                JsonValueKind.String => prop.Value.GetString(),
                JsonValueKind.Number => prop.Value.TryGetInt64(out var l) ? l : prop.Value.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null
            };
        }

        return dict;
    }

    private static IEnumerable<object> EnumerateJsonArray(JsonElement element)
    {
        foreach (var item in element.EnumerateArray())
        {
            yield return item;
        }
    }

    private static bool HasMultimodalContent(Dictionary<string, object?> payload)
    {
        if (!payload.TryGetValue("messages", out var messagesObj) || messagesObj is not IEnumerable<object> messages)
        {
            return false;
        }

        foreach (var message in messages)
        {
            if (message is not IDictionary<string, object?> dict)
            {
                continue;
            }

            if (!dict.TryGetValue("content", out var contentObj) || contentObj is null)
            {
                continue;
            }

            if (ContainsMultimodalBlock(contentObj))
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsMultimodalBlock(object contentObj)
    {
        if (contentObj is JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in element.EnumerateArray())
                {
                    if (IsMultimodalContentPart(item))
                    {
                        return true;
                    }
                }
            }

            if (element.ValueKind == JsonValueKind.Object && IsMultimodalContentPart(element))
            {
                return true;
            }

            return false;
        }

        if (contentObj is IEnumerable<object> list)
        {
            foreach (var item in list)
            {
                if (IsMultimodalContentPart(item))
                {
                    return true;
                }
            }
        }

        if (contentObj is IDictionary<string, object?> dict)
        {
            return IsMultimodalContentPart(dict);
        }

        return false;
    }

    private static bool IsMultimodalContentPart(object part)
    {
        string? type = null;

        if (part is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty("type", out var typeProp))
            {
                type = typeProp.GetString();
            }
        }
        else if (part is IDictionary<string, object?> dict && dict.TryGetValue("type", out var typeObj))
        {
            type = typeObj?.ToString();
        }

        if (string.IsNullOrWhiteSpace(type))
        {
            return false;
        }

        var lowered = type.ToLowerInvariant();
        return lowered is "image" or "input_image" or "document" or "input_document" or "input_file";
    }

    private static bool IsResponsesModel(Dictionary<string, object?> payload)
    {
        if (!payload.TryGetValue("model", out var modelObj) || modelObj is not string modelText)
        {
            return false;
        }

        var normalized = NormalizeResponsesModel(modelText).ToLowerInvariant();
        return normalized.Contains("gpt-5", StringComparison.OrdinalIgnoreCase) ||
               normalized.Contains("o3", StringComparison.OrdinalIgnoreCase);
    }

    private static Dictionary<string, object?> BuildStreamChunkFromAnthropic(MessagesResponse response)
    {
        var (contentText, toolCalls) = ExtractAnthropicContent(response.Content);
        var delta = new Dictionary<string, object?>
        {
            ["content"] = contentText
        };

        if (toolCalls.Count > 0)
        {
            delta["tool_calls"] = toolCalls;
        }

        return new Dictionary<string, object?>
        {
            ["choices"] = new[]
            {
                new Dictionary<string, object?>
                {
                    ["delta"] = delta,
                    ["finish_reason"] = MapAnthropicStopReason(response.StopReason)
                }
            },
            ["usage"] = new Dictionary<string, object?>
            {
                ["prompt_tokens"] = response.Usage.InputTokens,
                ["completion_tokens"] = response.Usage.OutputTokens
            }
        };
    }

    private static (string Text, List<Dictionary<string, object?>> ToolCalls) ExtractAnthropicContent(
        List<Dictionary<string, object?>> content)
    {
        var textBuilder = new StringBuilder();
        var toolCalls = new List<Dictionary<string, object?>>();

        foreach (var block in content)
        {
            if (!block.TryGetValue("type", out var typeObj))
            {
                continue;
            }

            var type = typeObj?.ToString();
            if (string.Equals(type, "text", StringComparison.OrdinalIgnoreCase))
            {
                if (block.TryGetValue("text", out var textObj))
                {
                    textBuilder.Append(textObj?.ToString());
                }
                continue;
            }

            if (string.Equals(type, "tool_use", StringComparison.OrdinalIgnoreCase))
            {
                var id = block.TryGetValue("id", out var idObj) ? idObj?.ToString() : null;
                var name = block.TryGetValue("name", out var nameObj) ? nameObj?.ToString() : null;
                var input = block.TryGetValue("input", out var inputObj) ? inputObj : null;

                if (!string.IsNullOrWhiteSpace(name))
                {
                    toolCalls.Add(new Dictionary<string, object?>
                    {
                        ["id"] = id ?? $"tool_{Guid.NewGuid()}" ,
                        ["function"] = new Dictionary<string, object?>
                        {
                            ["name"] = name,
                            ["arguments"] = input is null ? "{}" : JsonSerializer.Serialize(input)
                        }
                    });
                }
            }
        }

        return (textBuilder.ToString(), toolCalls);
    }

    private static string MapAnthropicStopReason(string? stopReason)
    {
        return stopReason switch
        {
            "max_tokens" => "length",
            "tool_use" => "tool_calls",
            _ => "stop"
        };
    }

    private static Dictionary<string, object?> ConvertChatResponse(ChatCompletion response)
    {
        var contentText = string.Concat(response.Content.Select(part => part.Text));
        var message = new Dictionary<string, object?>
        {
            ["content"] = contentText,
            ["tool_calls"] = BuildToolCalls(response)
        };

        return new Dictionary<string, object?>
        {
            ["id"] = response.Id,
            ["choices"] = new[]
            {
                new Dictionary<string, object?>
                {
                    ["finish_reason"] = response.FinishReason.ToString().ToLowerInvariant(),
                    ["message"] = message
                }
            },
            ["usage"] = new Dictionary<string, object?>
            {
                ["prompt_tokens"] = response.Usage?.InputTokenCount ?? 0,
                ["completion_tokens"] = response.Usage?.OutputTokenCount ?? 0
            }
        };
    }

    private static List<Dictionary<string, object?>>? BuildToolCalls(ChatCompletion response)
    {
        if (response.ToolCalls is null || response.ToolCalls.Count == 0)
        {
            return null;
        }

        var toolCalls = new List<Dictionary<string, object?>>();
        foreach (var toolCall in response.ToolCalls)
        {
            toolCalls.Add(new Dictionary<string, object?>
            {
                ["id"] = toolCall.Id,
                ["function"] = new Dictionary<string, object?>
                {
                    ["name"] = toolCall.FunctionName,
                    ["arguments"] = toolCall.FunctionArguments.ToString()
                }
            });
        }

        return toolCalls;
    }

    private static string ExtractDeployment(Dictionary<string, object?> payload)
    {
        if (payload.TryGetValue("model", out var modelObj) && modelObj is string modelText)
        {
            if (modelText.StartsWith("azure/", StringComparison.OrdinalIgnoreCase))
            {
                var trimmed = modelText["azure/".Length..];
                if (!string.IsNullOrWhiteSpace(trimmed))
                {
                    return trimmed;
                }
            }

            return modelText;
        }

        return string.Empty;
    }

    private static IReadOnlyList<ChatMessage> BuildChatMessages(Dictionary<string, object?> payload)
    {
        if (!payload.TryGetValue("messages", out var messagesObj) || messagesObj is not IEnumerable<object> messages)
        {
            return Array.Empty<ChatMessage>();
        }

        var list = new List<ChatMessage>();
        foreach (var message in messages)
        {
            if (message is not IDictionary<string, object?> dict)
            {
                continue;
            }

            var role = dict.TryGetValue("role", out var roleObj) ? roleObj?.ToString() : "user";
            var content = dict.TryGetValue("content", out var contentObj) ? contentObj?.ToString() ?? string.Empty : string.Empty;
            var toolCallId = dict.TryGetValue("tool_call_id", out var toolCallIdObj) ? toolCallIdObj?.ToString() : null;

            if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase))
            {
                var assistantMessage = new AssistantChatMessage(content);

                // 关键：保留历史 tool_calls。
                // Claude Code 会把上一次 assistant 的 tool_use + 本次 user 的 tool_result 一起发过来。
                // 若这里丢掉 assistant.tool_calls，Azure/OpenAI 侧会认为 tool 消息没有对应的前置 tool call。
                if (dict.TryGetValue("tool_calls", out var toolCallsObj) &&
                    toolCallsObj is not null &&
                    TryConvertToolCallsToChatToolCalls(toolCallsObj, out var toolCalls) &&
                    toolCalls.Count > 0)
                {
                    foreach (var toolCall in toolCalls)
                    {
                        assistantMessage.ToolCalls.Add(toolCall);
                    }
                }

                list.Add(assistantMessage);
                continue;
            }

            list.Add(role switch
            {
                "system" => new SystemChatMessage(content),
                "tool" when !string.IsNullOrWhiteSpace(toolCallId) => new ToolChatMessage(toolCallId, content),
                _ => new UserChatMessage(content)
            });
        }

        return list;
    }

    private static bool TryConvertToolCallsToChatToolCalls(object? toolCallsObj, out List<ChatToolCall> toolCalls)
    {
        toolCalls = new List<ChatToolCall>();

        if (toolCallsObj is null)
        {
            return false;
        }

        IEnumerable<object> toolCallItems = toolCallsObj switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.Array => EnumerateJsonArray(element),
            IEnumerable<object> list => list,
            _ => new[] { toolCallsObj }
        };

        foreach (var toolCall in toolCallItems)
        {
            Dictionary<string, object?>? dict = toolCall switch
            {
                JsonElement element when element.ValueKind == JsonValueKind.Object => ConvertJsonObjectToDictionary(element),
                IDictionary<string, object?> map => new Dictionary<string, object?>(map),
                _ => null
            };

            if (dict is null)
            {
                continue;
            }

            var id = ExtractToolCallId(dict) ?? $"tool_{Guid.NewGuid():N}";
            var name = ExtractToolCallName(dict) ?? string.Empty;
            var arguments = ExtractToolCallArguments(dict);

            if (string.IsNullOrWhiteSpace(name))
            {
                continue;
            }

            // OpenAI chat.completions expects stringified JSON for function arguments.
            toolCalls.Add(ChatToolCall.CreateFunctionToolCall(id, name, BinaryData.FromString(arguments)));
        }

        return toolCalls.Count > 0;
    }

    private static void NormalizeOpenAiMessages(Dictionary<string, object?> payload)
    {
        if (!payload.TryGetValue("messages", out var messagesObj) || messagesObj is not IEnumerable<object> messages)
        {
            return;
        }

        foreach (var message in messages)
        {
            if (message is not IDictionary<string, object?> dict)
            {
                continue;
            }

            if (!dict.TryGetValue("content", out var contentObj))
            {
                continue;
            }

            switch (contentObj)
            {
                case null:
                    dict["content"] = "...";
                    break;
                case JsonElement element when element.ValueKind == JsonValueKind.Array:
                    dict["content"] = element.GetRawText();
                    break;
                case IEnumerable<object> list:
                    dict["content"] = JsonSerializer.Serialize(list);
                    break;
            }

            var keysToRemove = dict.Keys
                .Where(key => key is not ("role" or "content" or "name" or "tool_call_id" or "tool_calls"))
                .ToList();
            foreach (var key in keysToRemove)
            {
                dict.Remove(key);
            }
        }
    }

    private static ChatCompletionOptions BuildChatOptions(Dictionary<string, object?> payload)
    {
        var options = new ChatCompletionOptions();

        if (payload.TryGetValue("max_tokens", out var maxTokens) && int.TryParse(maxTokens?.ToString(), out var max))
        {
            options.MaxOutputTokenCount = max;
        }

        if (payload.TryGetValue("temperature", out var temperature) && float.TryParse(temperature?.ToString(), out var temp))
        {
            options.Temperature = temp;
        }

        if (payload.TryGetValue("top_p", out var topP) && float.TryParse(topP?.ToString(), out var topValue))
        {
            options.TopP = topValue;
        }

        if (payload.TryGetValue("stop", out var stopObj) && stopObj is IEnumerable<object> stopList)
        {
            foreach (var item in stopList)
            {
                if (item is string stopString)
                {
                    options.StopSequences.Add(stopString);
                }
            }
        }

        if (payload.TryGetValue("tools", out var toolsObj) && toolsObj is IEnumerable<object> toolList)
        {
            foreach (var tool in toolList)
            {
                if (tool is IDictionary<string, object?> toolDict &&
                    toolDict.TryGetValue("function", out var functionObj) &&
                    functionObj is IDictionary<string, object?> functionDict &&
                    functionDict.TryGetValue("name", out var nameObj))
                {
                    var description = functionDict.TryGetValue("description", out var descObj)
                        ? descObj?.ToString()
                        : null;
                    var parameters = functionDict.TryGetValue("parameters", out var paramObj) && paramObj is JsonElement paramEl
                        ? paramEl
                        : paramObj ?? new Dictionary<string, object?>();

                    options.Tools.Add(ChatTool.CreateFunctionTool(
                        nameObj?.ToString() ?? string.Empty,
                        description,
                        BinaryData.FromObjectAsJson(parameters)));
                }
            }
        }

        if (payload.TryGetValue("tool_choice", out var toolChoiceObj))
        {
            options.ToolChoice = toolChoiceObj switch
            {
                string choice when choice == "auto" => ChatToolChoice.CreateAutoChoice(),
                string choice when choice == "any" => ChatToolChoice.CreateRequiredChoice(),
                string choice when choice == "none" => ChatToolChoice.CreateNoneChoice(),
                IDictionary<string, object?> dict => BuildToolChoice(dict),
                JsonElement element when element.ValueKind == JsonValueKind.Object => BuildToolChoice(element),
                _ => options.ToolChoice
            };
        }
        

        return options;
    }

    private static ChatToolChoice BuildToolChoice(IDictionary<string, object?> dict)
    {
        if (dict.TryGetValue("name", out var directName) && !string.IsNullOrWhiteSpace(directName?.ToString()))
        {
            return ChatToolChoice.CreateFunctionChoice(directName.ToString()!);
        }

        if (dict.TryGetValue("function", out var functionObj))
        {
            var name = ExtractFunctionName(functionObj);
            if (!string.IsNullOrWhiteSpace(name))
            {
                return ChatToolChoice.CreateFunctionChoice(name);
            }
        }

        return ChatToolChoice.CreateAutoChoice();
    }

    private static ChatToolChoice BuildToolChoice(JsonElement element)
    {
        if (element.TryGetProperty("name", out var nameDirect) && nameDirect.ValueKind == JsonValueKind.String)
        {
            var name = nameDirect.GetString();
            if (!string.IsNullOrWhiteSpace(name))
            {
                return ChatToolChoice.CreateFunctionChoice(name);
            }
        }

        if (element.TryGetProperty("function", out var functionProp) &&
            functionProp.ValueKind == JsonValueKind.Object &&
            functionProp.TryGetProperty("name", out var nameProp))
        {
            var name = nameProp.GetString();
            if (!string.IsNullOrWhiteSpace(name))
            {
                return ChatToolChoice.CreateFunctionChoice(name);
            }
        }

        return ChatToolChoice.CreateAutoChoice();
    }

    private static string? ExtractFunctionName(object? functionObj)
    {
        if (functionObj is JsonElement element && element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty("name", out var nameProp))
        {
            return nameProp.GetString();
        }

        if (functionObj is IDictionary<string, object?> dict && dict.TryGetValue("name", out var nameObj))
        {
            return nameObj?.ToString();
        }

        return null;
    }
}
