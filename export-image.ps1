param([string]$OutputPath = "$PSScriptRoot\clef-flash-api-0.1.0-linux-amd64.tar")
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputPath) { throw "Export already exists: $OutputPath" }
docker save -o $OutputPath iwaitu/clef-flash-api:latest clef-flash-api:0.1.0 clef-flash-api:latest
if ($LASTEXITCODE -ne 0) { throw 'Docker image export failed.' }
$archiveHash = Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256
"$($archiveHash.Hash.ToLowerInvariant())  $([System.IO.Path]::GetFileName($OutputPath))" | Set-Content -LiteralPath "$OutputPath.sha256" -Encoding ASCII
$archiveHash | Format-List
