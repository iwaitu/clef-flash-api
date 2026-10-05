$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath "$PSScriptRoot\config.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$fixtures = Get-Content -LiteralPath "$PSScriptRoot\test-fixtures.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$case = $fixtures.cases[2]
$payload = $case.request | ConvertTo-Json -Depth 30 -Compress
$client = [System.Net.Http.HttpClient]::new()
$client.BaseAddress = [Uri]"http://127.0.0.1:$($config.port)"
$client.Timeout = [TimeSpan]::FromSeconds(60)
function Measure-ClefHttp {
    $content = [System.Net.Http.StringContent]::new($payload, [System.Text.Encoding]::UTF8, 'application/json')
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $response = $client.PostAsync('/v1/systemone', $content).GetAwaiter().GetResult()
        try {
            $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            $clock.Stop()
            [void]$response.EnsureSuccessStatusCode()
            $body = $text | ConvertFrom-Json
            if ($body.answers.answer.choice -ne 'five') { throw 'Incorrect answer.' }
            [pscustomobject]@{http_ms=$clock.Elapsed.TotalMilliseconds; inference_ms=$body.timing.inference_ms; queue_ms=$body.timing.queue_ms}
        } finally { $response.Dispose() }
    } finally { $content.Dispose() }
}
try {
    1..3 | ForEach-Object { [void](Measure-ClefHttp) }
    $samples = @(1..30 | ForEach-Object { Measure-ClefHttp })
    $sorted = @($samples.http_ms | Sort-Object)
    $result = [ordered]@{
        client='Persistent .NET HttpClient, Windows host to Docker published port'
        case=$case.name
        requests=30
        correct=30
        http_ms=[ordered]@{p50=($sorted[14]+$sorted[15])/2; p95=$sorted[27]+($sorted[28]-$sorted[27])*0.55; min=$sorted[0]; max=$sorted[29]}
        samples=$samples
    }
    $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$PSScriptRoot\windows-http-test-results.json" -Encoding UTF8
    $result.http_ms | Format-List
} finally { $client.Dispose() }
