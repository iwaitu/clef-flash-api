$ErrorActionPreference = 'Stop'
docker build --platform linux/amd64 -t iwaitu/clef-flash-api:latest -t clef-flash-api:0.1.1 -t clef-flash-api:latest $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }
