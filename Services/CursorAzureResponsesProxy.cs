using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using AzureGptProxy.Infrastructure;

namespace AzureGptProxy.Services;

public sealed class CursorAzureResponsesProxy
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly NormalizedAzureOpenAiOptions _azureOptions;
    private readonly AzureOpenAiOptions _rawOptions;
    private readonly ILogger<CursorAzureResponsesProxy> _logger;

    public CursorAzureResponsesProxy(
        IHttpClientFactory httpClientFactory,
        NormalizedAzureOpenAiOptions azureOptions,
        Microsoft.Extensions.Options.IOptions<AzureOpenAiOptions> rawOptions,
        ILogger<CursorAzureResponsesProxy> logger)
    {
        _httpClientFactory = httpClientFactory;
        _azureOptions = azureOptions;
        _rawOptions = rawOptions.Value;
        _logger = logger;
    }

    public async Task<Stream> SendStreamingAsync(JsonDocument responsesRequestBody, CancellationToken cancellationToken)
    {
        var endpoint = _azureOptions.ResponsesEndpoint ?? _azureOptions.Endpoint;
        if (string.IsNullOrWhiteSpace(endpoint))
        {
            throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is required.");
        }

        if (string.IsNullOrWhiteSpace(_rawOptions.ApiKey))
        {
            throw new InvalidOperationException("AZURE_OPENAI_API_KEY is required.");
        }

        var httpClient = _httpClientFactory.CreateClient(nameof(CursorAzureResponsesProxy));

        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
        request.Headers.TryAddWithoutValidation("api-key", _rawOptions.ApiKey);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        if (_logger.IsEnabled(LogLevel.Debug))
        {
            _logger.LogDebug("cursor_azure_responses_raw_request {Request}", responsesRequestBody.RootElement.GetRawText());
        }

        request.Content = new StringContent(
            responsesRequestBody.RootElement.GetRawText(),
            Encoding.UTF8,
            "application/json");

        var response = await httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new InvalidOperationException(
                $"Azure Responses streaming request failed: {(int)response.StatusCode} {response.ReasonPhrase}. {body}");
        }

        _logger.LogInformation("Cursor Azure responses stream started status={StatusCode}", (int)response.StatusCode);

        // Caller must dispose the stream when done; the HttpResponseMessage will be disposed when stream is disposed.
        return new ResponseStreamWrapper(response);
    }

    private sealed class ResponseStreamWrapper : Stream
    {
        private readonly HttpResponseMessage _response;
        private readonly Stream _inner;

        public ResponseStreamWrapper(HttpResponseMessage response)
        {
            _response = response;
            _inner = response.Content.ReadAsStream();
        }

        public override bool CanRead => _inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override void Flush() => _inner.Flush();
        public override Task FlushAsync(CancellationToken cancellationToken) => _inner.FlushAsync(cancellationToken);

        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            _inner.ReadAsync(buffer, cancellationToken);

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            _inner.ReadAsync(buffer, offset, count, cancellationToken);

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _inner.Dispose();
                _response.Dispose();
            }
            base.Dispose(disposing);
        }

        public override async ValueTask DisposeAsync()
        {
            await _inner.DisposeAsync();
            _response.Dispose();
            await base.DisposeAsync();
        }
    }
}
