using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.Options;

namespace ClaudeAzureGptProxy.Infrastructure;

public sealed class AzureOpenAiOptions
{
    public string? Endpoint { get; set; }
    public string? ApiKey { get; set; }
    public string? ApiVersion { get; set; }
    public string? AuthToken { get; set; }
    public string? BigModel { get; set; }
    public string? SmallModel { get; set; }
}

public sealed class NormalizedAzureOpenAiOptions
{
    public string? Endpoint { get; set; }
    public string? ApiVersion { get; set; }
    public string? AuthToken { get; set; }
    public string? BigModel { get; set; }
    public string? SmallModel { get; set; }
    public string? ResponsesEndpoint { get; set; }
}

public static class AzureOpenAiConfig
{
    public static IServiceCollection AddAzureOpenAiConfig(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        services.Configure<AzureOpenAiOptions>(options =>
        {
            options.Endpoint = configuration["AZURE_OPENAI_ENDPOINT"];
            options.ApiKey = configuration["AZURE_OPENAI_API_KEY"];
            options.ApiVersion = configuration["AZURE_API_VERSION"];
            options.AuthToken = configuration["ANTHROPIC_AUTH_TOKEN"];
            options.BigModel = configuration["BIG_MODEL"];
            options.SmallModel = configuration["SMALL_MODEL"];
        });

        services.AddSingleton(provider =>
        {
            var options = provider.GetRequiredService<IOptions<AzureOpenAiOptions>>().Value;
            return NormalizeAzureOpenAiOptions(options);
        });

        return services;
    }

    public static NormalizedAzureOpenAiOptions NormalizeAzureOpenAiOptions(AzureOpenAiOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.Endpoint))
        {
            return new NormalizedAzureOpenAiOptions
            {
                Endpoint = options.Endpoint,
                ApiVersion = options.ApiVersion,
                AuthToken = options.AuthToken,
                BigModel = options.BigModel,
                SmallModel = options.SmallModel,
                ResponsesEndpoint = options.Endpoint,
            };
        }

        if (!Uri.TryCreate(options.Endpoint, UriKind.Absolute, out var parsed) ||
            string.IsNullOrWhiteSpace(parsed.Scheme) ||
            string.IsNullOrWhiteSpace(parsed.Host))
        {
            return new NormalizedAzureOpenAiOptions
            {
                Endpoint = options.Endpoint,
                ApiVersion = options.ApiVersion,
                AuthToken = options.AuthToken,
                BigModel = options.BigModel,
                SmallModel = options.SmallModel,
                ResponsesEndpoint = options.Endpoint,
            };
        }

        var normalizedEndpoint = $"{parsed.Scheme}://{parsed.Host}";
        var query = QueryHelpers.ParseQuery(parsed.Query);
        var queryVersion = query.TryGetValue("api-version", out var value) ? value.ToString() : null;
        var effectiveVersion = string.IsNullOrWhiteSpace(options.ApiVersion) ? queryVersion : options.ApiVersion;

        var responsesPath = parsed.AbsolutePath;
        if (string.IsNullOrWhiteSpace(responsesPath) ||
            !responsesPath.Contains("/openai/responses", StringComparison.OrdinalIgnoreCase))
        {
            responsesPath = "/openai/responses";
        }

        var responsesEndpoint = normalizedEndpoint.TrimEnd('/') + responsesPath;

        if (!string.IsNullOrWhiteSpace(effectiveVersion))
        {
            normalizedEndpoint = QueryHelpers.AddQueryString(normalizedEndpoint, "api-version", effectiveVersion);
            responsesEndpoint = QueryHelpers.AddQueryString(responsesEndpoint, "api-version", effectiveVersion);
        }

        return new NormalizedAzureOpenAiOptions
        {
            Endpoint = normalizedEndpoint,
            ApiVersion = effectiveVersion,
            AuthToken = options.AuthToken,
            BigModel = options.BigModel,
            SmallModel = options.SmallModel,
            ResponsesEndpoint = responsesEndpoint,
        };
    }
}
