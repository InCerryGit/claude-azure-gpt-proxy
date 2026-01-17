using System.Diagnostics;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Linq;
using ClaudeAzureGptProxy.Infrastructure;
using ClaudeAzureGptProxy.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenAI.Chat;
using OpenAI;

namespace ClaudeAzureGptProxy.Services;

public sealed class AzureOpenAiProxy
{
    private readonly AzureOpenAiClientFactory _clientFactory;
    private readonly ILogger<AzureOpenAiProxy> _logger;
    private readonly NormalizedAzureOpenAiOptions _azureOptions;
    private readonly AzureOpenAiOptions _rawOptions;

    public AzureOpenAiProxy(
        AzureOpenAiClientFactory clientFactory,
        ILogger<AzureOpenAiProxy> logger,
        NormalizedAzureOpenAiOptions azureOptions,
        IOptions<AzureOpenAiOptions> rawOptions)
    {
        _clientFactory = clientFactory;
        _logger = logger;
        _azureOptions = azureOptions;
        _rawOptions = rawOptions.Value;
    }

    public async Task<object> SendAsync(MessagesRequest request, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var payload = AnthropicConversion.ConvertAnthropicToAzure(request, _logger, _azureOptions);
        request.ResolvedAzureModel ??= payload["model"]?.ToString();
        NormalizeOpenAiMessages(payload);

        var isResponses = IsResponsesModel(payload);
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

        return ConvertChatResponse(response.Value);
    }

    public async IAsyncEnumerable<Dictionary<string, object?>> StreamAsync(
        MessagesRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var payload = AnthropicConversion.ConvertAnthropicToAzure(request, _logger, _azureOptions);
        request.ResolvedAzureModel ??= payload["model"]?.ToString();
        NormalizeOpenAiMessages(payload);

        var isResponses = IsResponsesModel(payload);
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
            var responsePayload = await SendResponsesAsync(payload, cancellationToken);
            _logger.LogInformation("Azure responses stream synth completed elapsedMs={ElapsedMs}", stopwatch.ElapsedMilliseconds);
            var synthesized = AnthropicConversion.ConvertAzureToAnthropic(responsePayload, request, _logger);
            var chunk = BuildStreamChunkFromAnthropic(synthesized);
            yield return chunk;
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

        using var httpClient = new HttpClient();
        using var requestMessage = new HttpRequestMessage(HttpMethod.Post, endpoint);
        requestMessage.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _rawOptions.ApiKey);
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
        return requestPayload;
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
                ["role"] = role,
                ["content"] = NormalizeResponsesContent(role, message.TryGetValue("content", out var contentObj) ? contentObj : null)
            };

            normalized.Add(normalizedMessage);
        }

        return normalized;
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
                : lastFunctionCallId;
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
            var copy = new Dictionary<string, object?>(partDict);
            var type = copy.TryGetValue("type", out var typeObj) ? typeObj?.ToString() : null;
            copy["type"] = NormalizeContentType(role, type) ?? DefaultContentTypeForRole(role);
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
            _ when lowered.StartsWith("input_") => lowered,
            _ => null
        };
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

            list.Add(role switch
            {
                "system" => new SystemChatMessage(content),
                "assistant" => new AssistantChatMessage(content),
                "tool" when !string.IsNullOrWhiteSpace(toolCallId) => new ToolChatMessage(toolCallId, content),
                _ => new UserChatMessage(content)
            });
        }

        return list;
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
