param(
    [string]$EnvFile = ".env.compose",
    [switch]$AllScenarios,
    [switch]$SkipInit,
    [switch]$NoBuild,
    [int]$HealthTimeoutSeconds = 420
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $RepoRoot $Path
}

function Ensure-Directory {
    param([string]$Path)
    $resolvedPath = Resolve-RepoPath $Path
    if (Test-Path -LiteralPath $resolvedPath) {
        $item = Get-Item -LiteralPath $resolvedPath
        if (-not $item.PSIsContainer) {
            throw "$Path must be a directory, but a file already exists at $resolvedPath."
        }
        return
    }
    New-Item -ItemType Directory -Path $resolvedPath -Force | Out-Null
}

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Name
    )
    $escapedName = [regex]::Escape($Name)
    $line = Get-Content -LiteralPath $Path |
        Where-Object { $_ -match "^\s*$escapedName\s*=" } |
        Select-Object -Last 1
    if (-not $line) {
        return ""
    }
    return (($line -replace "^\s*$escapedName\s*=", "").Trim().Trim('"').Trim("'"))
}

function Assert-ConfiguredValue {
    param(
        [string]$Path,
        [string]$Name,
        [int]$MinLength = 8
    )
    $value = Get-EnvValue -Path $Path -Name $Name
    if ([string]::IsNullOrWhiteSpace($value) -or
        $value.Length -lt $MinLength -or
        $value -match '[\u4e00-\u9fff]' -or
        $value -match '(?i)replace|changeme|change-me|your-|placeholder') {
        throw "$Name is not configured in $Path. Please edit the env file before deployment."
    }
}

function Invoke-Compose {
    param([string[]]$Arguments)
    & docker compose --env-file $EnvFile @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose command failed: $($Arguments -join ' ')"
    }
}

function Wait-ComposeHealth {
    param(
        [string]$Service,
        [int]$TimeoutSeconds
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $containerId = (& docker compose --env-file $EnvFile ps -q $Service).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect service: $Service"
        }
        if ($containerId) {
            $status = (& docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' $containerId 2>$null).Trim()
            if ($status -eq "healthy" -or $status -eq "running") {
                Write-Host "$Service is $status."
                return
            }
            Write-Host "$Service status: $status. Waiting..."
        }
        Start-Sleep -Seconds 5
    }
    throw "$Service did not become healthy within $TimeoutSeconds seconds."
}

$EnvFilePath = Resolve-RepoPath $EnvFile
if (-not (Test-Path -LiteralPath $EnvFilePath)) {
    Copy-Item -LiteralPath (Join-Path $RepoRoot ".env.compose.example") -Destination $EnvFilePath
    throw "Created $EnvFile. Please fill DASHSCOPE_API_KEY and ADMIN_API_TOKEN, then rerun this script."
}

Assert-ConfiguredValue -Path $EnvFilePath -Name "DASHSCOPE_API_KEY" -MinLength 12
Assert-ConfiguredValue -Path $EnvFilePath -Name "ADMIN_API_TOKEN" -MinLength 12

$env:ENV_FILE = $EnvFile

foreach ($directory in @("logs", "reports")) {
    Ensure-Directory -Path $directory
}

if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "site\index.html"))) {
    Write-Warning "site/index.html was not found. Run 'python -m mkdocs build' before opening /docs in Docker."
}

Write-Host "Validating docker compose config..."
Invoke-Compose -Arguments @("config", "--quiet")

Write-Host "Starting MySQL, etcd, MinIO and Milvus..."
Invoke-Compose -Arguments @("up", "-d", "mysql", "etcd", "minio", "milvus")
Wait-ComposeHealth -Service "mysql" -TimeoutSeconds $HealthTimeoutSeconds
Wait-ComposeHealth -Service "milvus" -TimeoutSeconds $HealthTimeoutSeconds

if (-not $NoBuild) {
    Write-Host "Building API image..."
    Invoke-Compose -Arguments @("build", "api")
}

if (-not $SkipInit) {
    if ($AllScenarios) {
        Write-Host "Initializing all scenarios..."
        Invoke-Compose -Arguments @("run", "--rm", "api", "python", "scripts/rebuild_scenarios.py", "--reset-collections")
    }
    else {
        $scenario = Get-EnvValue -Path $EnvFilePath -Name "ACTIVE_SCENARIO_ID"
        if ([string]::IsNullOrWhiteSpace($scenario)) {
            $scenario = "enterprise_knowledge"
        }
        Write-Host "Initializing active scenario: $scenario"
        Invoke-Compose -Arguments @(
            "run", "--rm", "api", "python", "scripts/rebuild_kb_version.py",
            "--scenario", $scenario,
            "--new-version", "--force", "--quality-gate", "--activate"
        )
    }
}

Write-Host "Starting API..."
Invoke-Compose -Arguments @("up", "-d", "api")
Invoke-Compose -Arguments @("ps")

$apiPort = Get-EnvValue -Path $EnvFilePath -Name "API_PORT"
if ([string]::IsNullOrWhiteSpace($apiPort)) {
    $apiPort = "8000"
}
Write-Host "KnowForge is ready at http://127.0.0.1:$apiPort/"
