using Microsoft.Extensions.Options;
using OpenAI;
using System.ClientModel;
using ClaudeAzureGptProxy.Infrastructure;

namespace ClaudeAzureGptProxy.Services;

public sealed class AzureOpenAiClientFactory
{
    private readonly AzureOpenAiOptions _options;
    private readonly NormalizedAzureOpenAiOptions _normalized;

    public AzureOpenAiClientFactory(IOptions<AzureOpenAiOptions> options, NormalizedAzureOpenAiOptions normalized)
    {
        _options = options.Value;
        _normalized = normalized;
    }

    public OpenAIClient CreateClient()
    {
        if (string.IsNullOrWhiteSpace(_normalized.Endpoint))
        {
            throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is required.");
        }

        if (string.IsNullOrWhiteSpace(_options.ApiKey))
        {
            throw new InvalidOperationException("AZURE_OPENAI_API_KEY is required.");
        }

        var endpoint = new Uri(_normalized.Endpoint, UriKind.Absolute);
        var credential = new ApiKeyCredential(_options.ApiKey);
        var clientOptions = new OpenAIClientOptions
        {
            Endpoint = endpoint
        };

        return new OpenAIClient(credential, clientOptions);
    }
}
