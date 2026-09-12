param(
    [string]$DataRoot = "C:\RAG-Agent-Data",
    [switch]$Refresh,
    [switch]$SkipRebuild,
    [string]$EmbeddingModel = "bge-m3"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$initScript = Join-Path $repoRoot "scripts\init-cybersec-kb.ps1"
$rebuildScript = Join-Path $repoRoot "scripts\rebuild-knowledge-base.ps1"
$importer = Join-Path $repoRoot "scripts\import_des_ctf_knowledge.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Project virtualenv Python was not found: $python" }
if (-not (Test-Path -LiteralPath $importer -PathType Leaf)) { throw "Importer was not found: $importer" }

function Invoke-GitChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed (exit $LASTEXITCODE): git $($Arguments -join ' ')"
    }
}

function Ensure-SourceCheckout {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url,
        [switch]$IncludePlainText
    )

    $sourceRoot = Join-Path (Join-Path $DataRoot "sources") $Name
    $gitRoot = Join-Path $sourceRoot ".git"
    $patterns = @("**/*.md", "**/*.markdown")
    if ($IncludePlainText) { $patterns += "**/*.txt" }
    $patterns += "/LICENSE"

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $sourceRoot) | Out-Null
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        Write-Host "Cloning $Name (text-only sparse checkout)..."
        Invoke-GitChecked @("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", $Url, $sourceRoot)
    } elseif (-not (Test-Path -LiteralPath $gitRoot -PathType Container)) {
        throw "Source root exists but is not a Git checkout: $sourceRoot"
    }

    $status = @(& git -C $sourceRoot status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect source checkout status: $sourceRoot" }
    # Sparse checkouts can report a deletion for an upstream file that cannot
    # be materialized on Windows; only reject modifications or untracked files.
    $unsafeStatus = @($status | Where-Object { $_ -notmatch '^\sD\s' })
    if ($unsafeStatus.Count -gt 0) {
        throw "Source checkout has local changes; refusing to alter it: $sourceRoot"
    }

    if ($Refresh) {
        Write-Host "Refreshing $Name..."
        Invoke-GitChecked @("-C", $sourceRoot, "fetch", "--depth", "1", "origin", "HEAD")
        Invoke-GitChecked @("-C", $sourceRoot, "reset", "--hard", "FETCH_HEAD")
    }

    Invoke-GitChecked @("-C", $sourceRoot, "sparse-checkout", "init", "--no-cone")
    $sparseArgs = @("-C", $sourceRoot, "sparse-checkout", "set", "--no-cone") + $patterns
    Invoke-GitChecked $sparseArgs
    Invoke-GitChecked @("-C", $sourceRoot, "read-tree", "-mu", "HEAD")

    $commit = (& git -C $sourceRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
        throw "Unable to read source commit: $sourceRoot"
    }
    [pscustomobject]@{ Name = $Name; Url = $Url; Root = $sourceRoot; Commit = $commit }
}

function Import-Source {
    param(
        [Parameter(Mandatory = $true)]$Source,
        [Parameter(Mandatory = $true)][string]$Category,
        [switch]$IncludePlainText,
        [string]$StripPrefix,
        [string[]]$LicenseFiles
    )

    $contentRoot = Join-Path $DataRoot "data\knowledge_base\cybersec\content\$Category"
    $metadataRoot = Join-Path $DataRoot "data\knowledge_base\cybersec\imports\$($Source.Name)"
    New-Item -ItemType Directory -Force -Path $contentRoot,$metadataRoot | Out-Null

    $args = @(
        $importer, "import",
        "--source-root", $Source.Root,
        "--destination-root", $contentRoot,
        "--metadata-root", $metadataRoot,
        "--commit", $Source.Commit,
        "--source-repository", $Source.Name,
        "--source-url", $Source.Url
    )
    if ($IncludePlainText) { $args += "--include-plain-text" }
    if ($StripPrefix) { $args += @("--strip-prefix", $StripPrefix) }
    foreach ($license in $LicenseFiles) { $args += @("--license-file", $license) }

    Write-Host "Importing $($Source.Name) at $($Source.Commit)..."
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Import failed: $($Source.Name)" }
}

# Keep local Ollama traffic direct if the shell has a proxy configured.
$noProxyHosts = @($env:NO_PROXY, $env:no_proxy, "127.0.0.1", "localhost") |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$env:NO_PROXY = (($noProxyHosts -join ',').Split(',') |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -Unique) -join ','
$env:no_proxy = $env:NO_PROXY

& $initScript -DataRoot $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Cybersecurity knowledge base initialization failed" }

$sources = @(
    @{ Name = "HackTricks"; Url = "https://github.com/HackTricks-wiki/hacktricks.git"; Category = "09_hacktricks"; IncludePlainText = $false; StripPrefix = "src"; LicenseFiles = @("src/LICENSE.md") },
    @{ Name = "PayloadsAllTheThings"; Url = "https://github.com/swisskyrepo/PayloadsAllTheThings.git"; Category = "10_payloads_all_the_things"; IncludePlainText = $true; StripPrefix = $null; LicenseFiles = @("LICENSE") }
)

foreach ($definition in $sources) {
    $source = Ensure-SourceCheckout -Name $definition.Name -Url $definition.Url -IncludePlainText:$definition.IncludePlainText
    Import-Source -Source $source -Category $definition.Category -IncludePlainText:$definition.IncludePlainText -StripPrefix $definition.StripPrefix -LicenseFiles $definition.LicenseFiles
}

if (-not $SkipRebuild) {
    Write-Host "Rebuilding cybersec vector index..."
    & $rebuildScript -DataRoot $DataRoot -EmbeddingModel $EmbeddingModel
    if ($LASTEXITCODE -ne 0) { throw "Cybersecurity KB rebuild failed" }
}

Write-Host "Security source import completed."
