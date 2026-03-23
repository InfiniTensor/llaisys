param(
    [ValidateSet("build", "test", "server", "all")]
    [string]$Mode = "all",
    [string]$Model = "",
    [string]$Device = "nvidia",
    [string]$CondaEnv = "llaisys-gpu",
    [string]$ConfigPath = "",
    [switch]$SkipTests,
    [switch]$ActivateConda
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "==> $Message"
}

$PythonExe = "python"
function Resolve-PythonExe {
    if ($env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return "python"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $RepoRoot "scripts\run_gpu.config.json"
}

$Config = $null
if (Test-Path $ConfigPath) {
    try {
        $Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    } catch {
        throw "Failed to read config file: $ConfigPath"
    }
}

if ($Config -ne $null) {
    if (-not $PSBoundParameters.ContainsKey("Model") -and $Config.model) {
        $Model = $Config.model
    }
    if (-not $PSBoundParameters.ContainsKey("Device") -and $Config.device) {
        $Device = $Config.device
    }
    if (-not $PSBoundParameters.ContainsKey("CondaEnv") -and $Config.conda_env) {
        $CondaEnv = $Config.conda_env
    }
}

if ($ActivateConda) {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        Write-Step "Activating conda env: $CondaEnv"
        conda activate $CondaEnv
    } else {
        throw "conda is not available in this shell. Run 'conda init powershell' and reopen PowerShell."
    }
}
$PythonExe = Resolve-PythonExe

function Build-Gpu {
    Write-Step "Configuring xmake"
    xmake f -m release --nv-gpu=y --vs=2022

    Write-Step "Building"
    xmake

    $dllSrc = Join-Path $RepoRoot "build\windows\x64\release\llaisys.dll"
    $dllDst = Join-Path $RepoRoot "python\llaisys\libllaisys\llaisys.dll"
    if (!(Test-Path $dllSrc)) {
        throw "Build output not found: $dllSrc"
    }

    Write-Step "Copying DLL to python package"
    Copy-Item $dllSrc $dllDst -Force
}

function Ensure-Dll {
    $dllDst = Join-Path $RepoRoot "python\llaisys\libllaisys\llaisys.dll"
    if (Test-Path $dllDst) {
        return
    }
    $dllCandidates = @(
        (Join-Path $RepoRoot "bin\llaisys.dll"),
        (Join-Path $RepoRoot "build\windows\x64\release\llaisys.dll")
    )
    foreach ($dllSrc in $dllCandidates) {
        if (Test-Path $dllSrc) {
            Write-Step "Copying DLL to python package"
            Copy-Item $dllSrc $dllDst -Force
            return
        }
    }
    throw "Missing llaisys.dll. Run '-Mode build' or copy it to: $dllDst"
}

function Test-Gpu {
    Ensure-Dll
    Write-Step "Running GPU op tests"
    & $PythonExe test/ops_gpu/run_all.py
}

function Run-Server {
    if ([string]::IsNullOrWhiteSpace($Model)) {
        throw "Model path is required. Provide -Model or set 'model' in $ConfigPath"
    }
    Ensure-Dll
    Write-Step "Starting server on $Device"
    & $PythonExe -m llaisys.server --model $Model --device $Device
}

switch ($Mode) {
    "build" { Build-Gpu }
    "test" { Test-Gpu }
    "server" { Run-Server }
    "all" {
        Build-Gpu
        if (-not $SkipTests) {
            Test-Gpu
        }
        Run-Server
    }
}

