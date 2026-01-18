using System.Diagnostics;
using System.Text.Json;
using ClaudeAzureGptProxy.Infrastructure;
using ClaudeAzureGptProxy.Models;
using ClaudeAzureGptProxy.Services;
using Microsoft.Extensions.DependencyInjection;
using Serilog;
using Serilog.Events;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.Services.AddAzureOpenAiConfig(builder.Configuration);
builder.Services.AddSingleton<AzureOpenAiClientFactory>();
builder.Services.AddHttpClient();
builder.Services.AddSingleton<AzureOpenAiProxy>();
builder.Services.AddSingleton<CursorAzureResponsesProxy>();
builder.Services.AddSingleton<TokenCounter>();
builder.Services.AddSingleton<ResponseLog>();

var logLevelSection = builder.Configuration.GetSection("Logging:LogLevel");
var defaultLevel = MapLogLevel(logLevelSection["Default"]) ?? LogEventLevel.Information;

var overrideLevels = logLevelSection.GetChildren()
    .Select(x => (Key: x.Key, Level: MapLogLevel(x.Value)))
    .Where(x => !string.IsNullOrWhiteSpace(x.Key) && !x.Key.Equals("Default", StringComparison.OrdinalIgnoreCase))
    .Where(x => x.Level is not null)
    .ToArray();

var logFilePath = Path.Combine(AppContext.BaseDirectory, "logs", "proxy-.log");
var logDirectory = Path.GetDirectoryName(logFilePath);
if (!string.IsNullOrWhiteSpace(logDirectory))
{
    Directory.CreateDirectory(logDirectory);
}

var loggerConfiguration = new LoggerConfiguration()
    .MinimumLevel.Is(defaultLevel)
    .Enrich.FromLogContext()
    .WriteTo.Console()
    .WriteTo.File(
        logFilePath,
        rollingInterval: RollingInterval.Day,
        retainedFileCountLimit: 7,
        fileSizeLimitBytes: 10 * 1024 * 1024,
        shared: true);

foreach (var (key, level) in overrideLevels)
{
    loggerConfiguration.MinimumLevel.Override(key, level!.Value);
}

Log.Logger = loggerConfiguration.CreateLogger();

builder.Host.UseSerilog();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.Lifetime.ApplicationStarted.Register(() =>
{
    var logger = app.Services.GetRequiredService<ILogger<Program>>();
    var envVarNames = Environment.GetEnvironmentVariables().Keys
        .Cast<object>()
        .Select(key => key?.ToString())
        .Where(name => !string.IsNullOrWhiteSpace(name))
        .OrderBy(name => name)
        .ToArray();
    logger.LogInformation("Environment variables: {EnvironmentVariables}", string.Join(", ", envVarNames));

    var urls = app.Urls.Count > 0 ? string.Join(", ", app.Urls) : builder.Configuration["ASPNETCORE_URLS"];
    if (string.IsNullOrWhiteSpace(urls))
    {
        urls = "unknown";
    }
    logger.LogInformation("Listening on: {Urls}", urls);
});

app.Use(async (context, next) =>
{
    var requiresAuth = context.Request.Path.StartsWithSegments("/v1/messages", StringComparison.OrdinalIgnoreCase)
                       || (context.Request.Path.StartsWithSegments("/cursor", StringComparison.OrdinalIgnoreCase)
                           && !context.Request.Path.StartsWithSegments("/cursor/health", StringComparison.OrdinalIgnoreCase));

    if (requiresAuth)
    {
        var azureOptions = context.RequestServices.GetRequiredService<NormalizedAzureOpenAiOptions>();
        var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
        if (!string.IsNullOrWhiteSpace(azureOptions.AuthToken))
        {
            var authHeader = context.Request.Headers.Authorization.ToString();
            if (string.IsNullOrWhiteSpace(authHeader))
            {
                logger.LogWarning("Authorization failed: missing Authorization header.");
                context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                await context.Response.WriteAsJsonAsync(new { error = "Unauthorized", message = "Missing Authorization header." });
                return;
            }

            if (!authHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                logger.LogWarning("Authorization failed: invalid scheme.");
                context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                await context.Response.WriteAsJsonAsync(new { error = "Unauthorized", message = "Invalid Authorization scheme." });
                return;
            }

            var token = authHeader["Bearer ".Length..].Trim();
            if (!string.Equals(token, azureOptions.AuthToken, StringComparison.Ordinal))
            {
                logger.LogWarning("Authorization failed: invalid token.");
                context.Response.StatusCode = StatusCodes.Status403Forbidden;
                await context.Response.WriteAsJsonAsync(new { error = "Forbidden", message = "Invalid auth token." });
                return;
            }
        }
    }

    await next();
});

app.MapGet("/cursor/health", () => Results.Json(new { status = "ok" }));

app.MapGet("/cursor/models", () =>
{
    var models = new[] { "gpt-high", "gpt-medium", "gpt-low", "gpt-minimal" };
    var response = new
    {
        @object = "list",
        data = models.Select(id => new { id, @object = "model", created = 0, owned_by = "cursor" }).ToArray()
    };
    return Results.Json(response);
});

app.MapGet("/cursor/v1/models", () =>
{
    var models = new[] { "gpt-high", "gpt-medium", "gpt-low", "gpt-minimal" };
    var response = new
    {
        @object = "list",
        data = models.Select(id => new { id, @object = "model", created = 0, owned_by = "cursor" }).ToArray()
    };
    return Results.Json(response);
});

app.MapPost("/cursor/chat/completions", HandleCursorChatCompletions);
app.MapPost("/cursor/v1/chat/completions", HandleCursorChatCompletions);


app.MapGet("/", () => Results.Json(new { message = "Anthropic Proxy for Azure OpenAI" }));

static async Task<IResult> HandleCursorChatCompletions(
    OpenAiChatCompletionsRequest request,
    CursorAzureResponsesProxy proxy,
    NormalizedAzureOpenAiOptions azureOptions,
    HttpResponse response,
    ILogger<Program> logger,
    CancellationToken cancellationToken)
{
    // python 行为：强制流式
    response.StatusCode = StatusCodes.Status200OK;
    response.ContentType = "text/event-stream";

    (JsonDocument responsesBody, string inboundModel) built;

    try
    {
        built = CursorRequestAdapter.BuildResponsesRequest(request, azureOptions);
    }
    catch (ArgumentException ex)
    {
        return Results.Problem(detail: ex.Message, statusCode: StatusCodes.Status400BadRequest, title: "Bad Request");
    }

    using var responsesBody = built.responsesBody;
    var inboundModel = built.inboundModel;

    // Cursor 兼容性问题排查：只在 Debug 级别输出最关键的上下文。
    // 生产环境将 Logging:LogLevel:Default / Microsoft.AspNetCore 提高即可自动关闭。
    if (logger.IsEnabled(LogLevel.Debug))
    {
        logger.LogDebug(
            "cursor_request model={Model} messages={MessageCount} tools={ToolCount} tool_choice_present={HasToolChoice} user_present={HasUser}",
            request.Model,
            request.Messages?.Count ?? 0,
            request.Tools?.Count ?? 0,
            request.ToolChoice.HasValue && request.ToolChoice.Value.ValueKind is not JsonValueKind.Undefined and not JsonValueKind.Null,
            !string.IsNullOrWhiteSpace(request.User));
    }

    await using var azureStream = await proxy.SendStreamingAsync(responsesBody, cancellationToken);
    var adapter = new CursorResponseAdapter(inboundModel);
    var decoder = new SseDecoder();

    using var reader = new StreamReader(azureStream);
    var debugSseCount = 0;
    const int debugSseMax = 30; // 只打前 N 条，避免日志刷屏
    while (!cancellationToken.IsCancellationRequested)
    {
        var line = await reader.ReadLineAsync(cancellationToken);
        if (line is null)
        {
            break;
        }

        if (logger.IsEnabled(LogLevel.Debug) && debugSseCount < debugSseMax)
        {
            var singleLine = line.Replace("\r", "\\r", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal);
            logger.LogDebug("cursor_upstream_line[{Index}] {Line}", debugSseCount, singleLine);
        }

        foreach (var data in decoder.PushLine(line))
        {
            foreach (var sse in adapter.ConvertAzureSseDataToOpenAiSse(data))
            {
                if (logger.IsEnabled(LogLevel.Debug) && debugSseCount < debugSseMax)
                {
                    // 记录下行 SSE（单行转义），用于诊断 Cursor "looping detected"。
                    var singleLine = sse.Replace("\r", "\\r", StringComparison.Ordinal)
                        .Replace("\n", "\\n", StringComparison.Ordinal);
                    logger.LogDebug("cursor_downstream_sse[{Index}] {Sse}", debugSseCount, singleLine);
                    debugSseCount++;
                }

                await response.WriteAsync(sse, cancellationToken);
                await response.Body.FlushAsync(cancellationToken);
            }
        }
    }

    // 兜底：确保以 [DONE] 结束（如果 Azure 没发 response.completed）
    await response.WriteAsync(OpenAiSseEncoder.Done(), cancellationToken);
    await response.Body.FlushAsync(cancellationToken);

    if (logger.IsEnabled(LogLevel.Debug))
    {
        logger.LogDebug("cursor_downstream_sse_done wrote_done=true printed_first_n={N}", debugSseCount);
    }

    logger.LogInformation("/cursor/chat/completions stream finished inboundModel={InboundModel}", inboundModel);
    return Results.Empty;
}


app.MapPost("/v1/messages", async (
    MessagesRequest request,
    AzureOpenAiProxy proxy,
    NormalizedAzureOpenAiOptions azureOptions,
    ILogger<Program> logger,
    ResponseLog responseLog,
    HttpResponse response,
    CancellationToken cancellationToken) =>
{
    var requestId = Guid.NewGuid().ToString("N");
    using var scope = logger.BeginScope(new Dictionary<string, object?>
    {
        ["requestId"] = requestId
    });

    var stopwatch = Stopwatch.StartNew();

    request.OriginalModel ??= request.Model;
    request.ResolvedAzureModel = AnthropicConversion.ResolveAzureModel(request, azureOptions);
    responseLog.LogRequest(request, isStream: request.Stream);

    logger.LogInformation("/v1/messages request start model {Model} resolved {ResolvedModel} stream={Stream} max_tokens={MaxTokens}",
        request.Model,
        request.ResolvedAzureModel ?? request.Model,
        request.Stream,
        request.MaxTokens);

    if (request.Stream)
    {
        response.StatusCode = StatusCodes.Status200OK;
        response.ContentType = "text/event-stream";

        var sseStats = new SseStreaming.StreamStats();
        var sseEventIndex = 0;

        try
        {
            var stream = proxy.StreamAsync(request, cancellationToken);
            await foreach (var sse in SseStreaming.HandleStreaming(stream, request, logger, sseStats)
                               .WithCancellation(cancellationToken))
            {
                responseLog.LogAnthropicSseEvent(sseEventIndex, sse);
                sseEventIndex++;

                await response.WriteAsync(sse, cancellationToken);
                await response.Body.FlushAsync(cancellationToken);
            }

            if (sseStats.AggregatedResponse is not null)
            {
                responseLog.LogAnthropicAggregatedResponse(sseStats.AggregatedResponse);
            }

            logger.LogInformation(
                "/v1/messages stream completed elapsedMs={ElapsedMs} chunkCount={ChunkCount} eventCount={EventCount} outputChars={OutputChars} inputTokens={InputTokens} outputTokens={OutputTokens}",
                stopwatch.ElapsedMilliseconds,
                sseStats.ChunkCount,
                sseStats.EventCount,
                sseStats.OutputCharacters,
                sseStats.InputTokens,
                sseStats.OutputTokens);
        }
        catch (OperationCanceledException)
        {
            logger.LogInformation(
                "Streaming canceled by client. elapsedMs={ElapsedMs} chunkCount={ChunkCount} eventCount={EventCount}",
                stopwatch.ElapsedMilliseconds,
                sseStats.ChunkCount,
                sseStats.EventCount);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Streaming error");
            if (!response.HasStarted)
            {
                response.StatusCode = StatusCodes.Status500InternalServerError;
            }
        }

        return Results.Empty;
    }

    try
    {
        var azureResponse = await proxy.SendAsync(request, cancellationToken);
        var anthropicResponse = AnthropicConversion.ConvertAzureToAnthropic(azureResponse, request, logger);
        responseLog.LogAnthropicResponse(anthropicResponse);
        logger.LogInformation(
            "/v1/messages request completed elapsedMs={ElapsedMs} stopReason={StopReason} inputTokens={InputTokens} outputTokens={OutputTokens}",
            stopwatch.ElapsedMilliseconds,
            anthropicResponse.StopReason ?? "(unknown)",
            anthropicResponse.Usage.InputTokens,
            anthropicResponse.Usage.OutputTokens);
        return Results.Json(anthropicResponse);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Request failed");
        return Results.Problem(
            detail: ex.Message,
            statusCode: StatusCodes.Status500InternalServerError,
            title: "Proxy request failed");
    }
});

app.MapPost("/v1/messages/count_tokens", (
    TokenCountRequest request,
    NormalizedAzureOpenAiOptions azureOptions,
    TokenCounter tokenCounter,
    HttpRequest httpRequest,
    ILogger<Program> logger) =>
{
    request.OriginalModel ??= request.Model;
    request.ResolvedAzureModel = AnthropicConversion.ResolveAzureModel(request, azureOptions);

    var inputTokens = tokenCounter.CountInputTokens(request);
    logger.LogDebug("/v1/messages/count_tokens raw request {Request}", JsonSerializer.Serialize(request));
    logger.LogInformation("/v1/messages/count_tokens request for model {Model} resolved {ResolvedModel} input_tokens={InputTokens}",
        request.Model,
        request.ResolvedAzureModel ?? request.Model,
        inputTokens);
    return Results.Json(new TokenCountResponse { InputTokens = inputTokens });
});

app.Run();

static LogEventLevel? MapLogLevel(string? value)
{
    if (string.IsNullOrWhiteSpace(value))
    {
        return null;
    }

    if (value.Equals("Trace", StringComparison.OrdinalIgnoreCase))
    {
        return LogEventLevel.Verbose;
    }

    if (value.Equals("Critical", StringComparison.OrdinalIgnoreCase))
    {
        return LogEventLevel.Fatal;
    }

    return Enum.TryParse(value, true, out LogEventLevel level) ? level : null;
}
