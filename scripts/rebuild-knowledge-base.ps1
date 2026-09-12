param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$KnowledgeBase,
    [string]$EmbeddingModel = "bge-m3",
    [string]$DataRoot = "C:\RAG-Agent-Data",
    [string]$ServerRoot = $env:CHATCHAT_SERVER_ROOT
)

# Rebuild a knowledge base index with the upstream LangGraph-Chatchat CLI, then
# refresh the standalone cosine sidecar this repository serves from.
#
# The index builder lives outside this repository, so its location must be given
# explicitly (or exported as CHATCHAT_SERVER_ROOT). This repository provides the
# retrieval service, not the embedding pipeline.
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($ServerRoot)) {
    throw @"
The upstream index builder was not specified.
Pass -ServerRoot <path to a chatchat-server checkout> or set:
  `$env:CHATCHAT_SERVER_ROOT = 'C:\path\to\LangGraph-Chatchat\chatchat-server'
"@
}
$ServerRoot = (Resolve-Path -LiteralPath $ServerRoot).Path
$builderPython = Join-Path $ServerRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $builderPython -PathType Leaf)) {
    throw "Index builder virtualenv Python was not found: $builderPython"
}
if (-not (Test-Path -LiteralPath $localPython -PathType Leaf)) {
    throw "Repository virtualenv Python was not found: $localPython"
}

$env:CHATCHAT_ROOT = $DataRoot
if ($EmbeddingModel -eq "embedding-3" -and [string]::IsNullOrWhiteSpace($env:ZAI_API_KEY)) {
    throw "ZAI_API_KEY is not set. Configure the Zhipu API key before rebuilding with embedding-3."
}
# Keep cloud API traffic direct unless the user explicitly configured a proxy.
$noProxyHosts = @($env:NO_PROXY, $env:no_proxy, "127.0.0.1", "localhost") |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$env:NO_PROXY = (($noProxyHosts -join ",").Split(",") |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -Unique) -join ","
$env:no_proxy = $env:NO_PROXY

Push-Location -LiteralPath $ServerRoot
$previousPythonWarnings = $env:PYTHONWARNINGS
$env:PYTHONWARNINGS = "ignore::DeprecationWarning"
try {
    & $builderPython "chatchat\cli.py" kb -r -n $KnowledgeBase -e $EmbeddingModel
    $exitCode = $LASTEXITCODE
} finally {
    if ($null -eq $previousPythonWarnings) {
        Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONWARNINGS = $previousPythonWarnings
    }
    Pop-Location
}
if ($exitCode -ne 0) {
    throw "Knowledge base rebuild failed with exit code $exitCode"
}

# The standalone RAG backend serves from the converted cosine sidecar, so it
# must be regenerated after every committed index rebuild or it will reject
# the index as stale. A failure here leaves the new index intact; re-run
# `python -m rag_service.build_cosine` to recover.
Write-Host "Refreshing standalone cosine sidecar for $KnowledgeBase..."
$kbRoot = Join-Path $DataRoot "data\knowledge_base"
& $localPython -m rag_service.build_cosine --kb-root $kbRoot --knowledge-base $KnowledgeBase
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Sidecar refresh failed; standalone retrieval will report the index as stale until "python -m rag_service.build_cosine" succeeds.'
}
