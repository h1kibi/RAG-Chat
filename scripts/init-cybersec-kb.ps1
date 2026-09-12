param(
    [string]$DataRoot = "C:\RAG-Agent-Data",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$templateRoot = Join-Path $repoRoot "knowledge-base\cybersec"
$helper = Join-Path $repoRoot "scripts\cybersec_kb.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtualenv Python was not found: $python. Create it with 'python -m venv .venv' and install requirements.txt."
}
if (-not (Test-Path -LiteralPath $templateRoot -PathType Container)) {
    throw "Cybersecurity template was not found: $templateRoot"
}

$helperArgs = @(
    $helper,
    "init",
    "--template-root", $templateRoot,
    "--data-root", $DataRoot
)
if ($Overwrite) { $helperArgs += "--overwrite" }

# The initializer is standard library only; no CHATCHAT_ROOT is required to
# copy the template into the runtime data root.
& $python @helperArgs
if ($LASTEXITCODE -ne 0) {
    throw "Cybersecurity knowledge base initialization failed with exit code $LASTEXITCODE"
}
