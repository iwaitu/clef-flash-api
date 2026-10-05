$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath "$PSScriptRoot\config.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$payload = Get-Content -LiteralPath "$PSScriptRoot\example-request.json" -Raw -Encoding UTF8
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$answer = Invoke-RestMethod -Uri "http://127.0.0.1:$($config.port)/v1/systemone" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($payload)) -TimeoutSec 60
$clock.Stop()
$answer | ConvertTo-Json -Depth 20
Write-Host ('Windows HTTP round trip: {0:N2} ms' -f $clock.Elapsed.TotalMilliseconds)
if ($answer.answers.answer.choice -ne 'five') { throw 'Unexpected answer to the apple question.' }
