$ErrorActionPreference = 'Stop'

function Test-CarlaInstallation([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    $shippingExe = Get-ChildItem -LiteralPath $Path -Recurse -File -Filter 'CarlaUE4-Win64-Shipping.exe' |
        Where-Object Length -gt 0 | Select-Object -First 1
    if (-not $shippingExe) {
        return $false
    }

    # CARLA can ship Town01 as a loose Unreal map or inside the main Windows package.
    $map = Get-ChildItem -LiteralPath $Path -Recurse -File -Filter 'Town01.umap' |
        Where-Object Length -gt 0 | Select-Object -First 1
    if ($map) {
        return $true
    }
    $package = Get-ChildItem -LiteralPath $Path -Recurse -File -Filter '*.pak' |
        Where-Object { $_.Length -gt 0 -and $_.Name -match '^(Town01|CarlaUE4-WindowsNoEditor|pakchunk0-WindowsNoEditor)\.pak$' } |
        Select-Object -First 1
    return [bool]$package
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (Test-Path -LiteralPath .venv) {
            throw 'The existing .venv has no Python executable. Move that folder aside before rerunning setup.'
        }
        if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
            throw 'Install Python 3.12 for Windows with the Python launcher, then rerun setup.'
        }
        Write-Host 'Creating the Python 3.12 environment.'
        & py.exe -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.12 environment.' }
    }

    & $python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'
    if ($LASTEXITCODE -ne 0) { throw 'The existing .venv must use Python 3.12. Move it aside before rerunning setup.' }
    & $python -m pip install --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the Python dependencies.' }

    if ($env:CARLA_ROOT) {
        if (-not (Test-CarlaInstallation $env:CARLA_ROOT)) {
            throw "CARLA_ROOT does not contain the Windows shipping executable and Town01 or its CARLA package: $env:CARLA_ROOT"
        }
        Write-Host "Using existing CARLA installation at $env:CARLA_ROOT."
        return
    }

    $local = Join-Path $PSScriptRoot '.local'
    $carla = Join-Path $local 'carla'
    if (Test-CarlaInstallation $carla) {
        Write-Host "Using existing CARLA installation at $carla."
        return
    }
    if (Test-Path -LiteralPath $carla) {
        throw "The existing CARLA directory is incomplete. Move it aside before rerunning setup: $carla"
    }

    New-Item -Path $local -ItemType Directory -Force | Out-Null
    $archive = Join-Path $local 'CARLA_0.9.16.zip'
    $archiveUrl = 'https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip'
    $expectedBytes = 7812252810L
    $archiveBytes = 0L
    if (Test-Path -LiteralPath $archive) {
        $archiveBytes = (Get-Item -LiteralPath $archive).Length
    }
    if ($archiveBytes -gt $expectedBytes) {
        throw "The existing archive is larger than the official download. Move it aside before rerunning setup: $archive"
    }
    if ($archiveBytes -lt $expectedBytes) {
        Write-Host 'Downloading CARLA 0.9.16 (7.8 GB). Rerun setup to resume an interrupted download.'
        & curl.exe --fail --location --retry 4 --continue-at - --output $archive $archiveUrl
        if ($LASTEXITCODE -ne 0) { throw "CARLA download interrupted. The partial archive is kept at $archive." }
    }
    if ((Get-Item -LiteralPath $archive).Length -ne $expectedBytes) {
        throw "CARLA archive is incomplete. Rerun setup to resume: $archive"
    }

    # Publish the directory only after extraction and validation succeed.
    $staging = Join-Path $local ('carla-extract-' + [guid]::NewGuid())
    New-Item -Path $staging -ItemType Directory | Out-Null
    Write-Host 'Extracting CARLA 0.9.16.'
    & tar.exe -xf $archive -C $staging
    if ($LASTEXITCODE -ne 0 -or -not (Test-CarlaInstallation $staging)) {
        throw "CARLA extraction did not produce a valid installation. Files are kept at $staging."
    }
    [IO.Directory]::Move($staging, $carla)
    Write-Host "CARLA 0.9.16 is installed at $carla."
}
finally {
    Pop-Location
}
