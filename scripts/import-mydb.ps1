param(
    [string]$MyDBRoot = "C:\Tools\MyDB",
    [string]$DataRoot = "C:\RAG-Agent-Data",
    [switch]$SkipRebuild,
    [string]$EmbeddingModel = "bge-m3",
    [string]$ServerRoot = $env:CHATCHAT_SERVER_ROOT
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$importer = Join-Path $repoRoot "scripts\import_des_ctf_knowledge.py"
$initScript = Join-Path $repoRoot "scripts\init-cybersec-kb.ps1"
$rebuildScript = Join-Path $repoRoot "scripts\rebuild-cybersec.ps1"

$MyDBRoot = [System.IO.Path]::GetFullPath($MyDBRoot)
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)

foreach ($requiredPath in @($python, $importer, $initScript, $rebuildScript)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file was not found: $requiredPath. Create the repository virtualenv and install requirements.txt."
    }
}
if (-not (Test-Path -LiteralPath $MyDBRoot -PathType Container)) {
    throw "MyDB root was not found: $MyDBRoot"
}

function Get-SourceRevision {
    param([Parameter(Mandatory = $true)][string]$SourceRoot)

    $gitRoot = Join-Path $SourceRoot ".git"
    if (Test-Path -LiteralPath $gitRoot -PathType Container) {
        $commit = (& git -C $SourceRoot rev-parse HEAD).Trim()
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($commit)) {
            return $commit
        }
    }

    $files = @(
        Get-ChildItem -LiteralPath $SourceRoot -File -Recurse -Force |
            Where-Object { $_.FullName -notmatch "[\\/]\.git[\\/]" }
    )
    if ($files.Count -eq 0) {
        return "local-empty"
    }
    $totalBytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    $latest = ($files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc.Ticks
    return "local-$($files.Count)-$totalBytes-$latest"
}

function Import-MyDBSource {
    param([Parameter(Mandatory = $true)]$Definition)

    $sourceRoot = Join-Path $MyDBRoot $Definition.Directory
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        throw "Configured MyDB source was not found: $sourceRoot"
    }

    $contentRoot = Join-Path $DataRoot "data\knowledge_base\cybersec\content\$($Definition.Category)"
    $metadataRoot = Join-Path $DataRoot "data\knowledge_base\cybersec\imports\$($Definition.RepositoryName)"
    New-Item -ItemType Directory -Force -Path $contentRoot,$metadataRoot | Out-Null

    $revision = Get-SourceRevision -SourceRoot $sourceRoot
    $importArgs = @(
        $importer, "import",
        "--source-root", $sourceRoot,
        "--destination-root", $contentRoot,
        "--metadata-root", $metadataRoot,
        "--commit", $revision,
        "--source-repository", $Definition.RepositoryName,
        "--source-url", $Definition.SourceUrl,
        "--no-provenance-header",
        "--strip-markdown-noise"
    )
    if ($Definition.IncludePlainText) { $importArgs += "--include-plain-text" }
    if ($Definition.IncludeStructuredText) { $importArgs += "--include-structured-text" }
    if (-not [string]::IsNullOrWhiteSpace($Definition.StripPrefix)) {
        $importArgs += @("--strip-prefix", $Definition.StripPrefix)
    }
    foreach ($licenseFile in $Definition.LicenseFiles) {
        $importArgs += @("--license-file", $licenseFile)
    }

    Write-Host "Importing $($Definition.RepositoryName) from $sourceRoot at $revision..."
    & $python @importArgs
    if ($LASTEXITCODE -ne 0) {
        throw "MyDB import failed: $($Definition.RepositoryName)"
    }
}

# Initialize curated documents and merge importer metadata without deleting user files.
& $initScript -DataRoot $DataRoot
if ($LASTEXITCODE -ne 0) {
    throw "Cybersecurity knowledge base initialization failed"
}

$definitions = @(
    [pscustomobject]@{
        Directory = "Des-CTF-Knowledge"
        RepositoryName = "Des-CTF-Knowledge"
        Category = "08_ctf_des_knowledge"
        SourceUrl = "local://MyDB/Des-CTF-Knowledge"
        IncludePlainText = $false
        IncludeStructuredText = $false
        StripPrefix = ""
        LicenseFiles = @("LICENSE", "LICENSE.notice")
    },
    [pscustomobject]@{
        Directory = "hacktricks"
        RepositoryName = "HackTricks"
        Category = "09_hacktricks"
        SourceUrl = "local://MyDB/hacktricks"
        IncludePlainText = $false
        IncludeStructuredText = $false
        StripPrefix = "src"
        LicenseFiles = @("src/LICENSE.md")
    },
    [pscustomobject]@{
        Directory = "PayloadsAllTheThings"
        RepositoryName = "PayloadsAllTheThings"
        Category = "10_payloads_all_the_things"
        SourceUrl = "local://MyDB/PayloadsAllTheThings"
        IncludePlainText = $true
        IncludeStructuredText = $false
        StripPrefix = ""
        LicenseFiles = @("LICENSE")
    },
    [pscustomobject]@{
        Directory = "LOLBAS"
        RepositoryName = "LOLBAS"
        Category = "11_lolbas"
        SourceUrl = "local://MyDB/LOLBAS"
        IncludePlainText = $false
        IncludeStructuredText = $true
        StripPrefix = ""
        LicenseFiles = @("LICENSE", "NOTICE.md")
    },
    [pscustomobject]@{
        Directory = "Security-Learning"
        RepositoryName = "Security-Learning"
        Category = "12_security_learning"
        SourceUrl = "local://MyDB/Security-Learning"
        IncludePlainText = $false
        IncludeStructuredText = $false
        StripPrefix = ""
        LicenseFiles = @()
    },
    [pscustomobject]@{
        Directory = "xianzhi"
        RepositoryName = "xianzhi"
        Category = "13_xianzhi"
        SourceUrl = "local://MyDB/xianzhi"
        IncludePlainText = $false
        IncludeStructuredText = $false
        StripPrefix = ""
        LicenseFiles = @()
    },
    [pscustomobject]@{
        Directory = "CTF-WP"
        RepositoryName = "CTF-WP"
        Category = "14_ctf_wp"
        SourceUrl = "local://MyDB/CTF-WP"
        IncludePlainText = $true
        IncludeStructuredText = $false
        StripPrefix = ""
        LicenseFiles = @()
    },
    [pscustomobject]@{
        Directory = "butian"
        RepositoryName = "butian"
        Category = "15_butian"
        SourceUrl = "local://MyDB/butian"
        IncludePlainText = $false
        IncludeStructuredText = $false
        StripPrefix = ""
        LicenseFiles = @()
    }
)

foreach ($definition in $definitions) {
    Import-MyDBSource -Definition $definition
}
if (-not $SkipRebuild) {
    Write-Host "Rebuilding cybersec vector index..."
    & $rebuildScript -DataRoot $DataRoot -EmbeddingModel $EmbeddingModel `
        -ServerRoot $ServerRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Cybersecurity KB rebuild failed"
    }
}

Write-Host "MyDB synchronization completed."
