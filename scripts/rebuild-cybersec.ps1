param(
    [string]$DataRoot = "C:\RAG-Agent-Data",
    [string]$EmbeddingModel = "bge-m3",
    [string]$ServerRoot = $env:CHATCHAT_SERVER_ROOT
)

$ErrorActionPreference = "Stop"
$initScript = Join-Path $PSScriptRoot "init-cybersec-kb.ps1"
$rebuildScript = Join-Path $PSScriptRoot "rebuild-knowledge-base.ps1"

& $initScript -DataRoot $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Cybersecurity KB initialization failed" }
& $rebuildScript -KnowledgeBase "cybersec" -EmbeddingModel $EmbeddingModel `
    -DataRoot $DataRoot -ServerRoot $ServerRoot
if ($LASTEXITCODE -ne 0) { throw "Cybersecurity KB rebuild failed" }
