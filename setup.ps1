param([switch]$ClientOnly)
$ErrorActionPreference = 'Stop'

$baseName = 'CARLA_1.0.zip'
$mapsName = 'AdditionalMaps_1.0.tar.gz'
$wheelName = 'carla-1.0-cp312-cp312-win_amd64.whl'
$local = Join-Path $PSScriptRoot '.local'

function Get-ReleaseArchive([string]$Name, [long]$ExpectedBytes) {
    $path = Join-Path $local $Name
    $size = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0L }
    if ($size -gt $ExpectedBytes) { throw "Unexpected size for $path. Preserve it separately before retrying." }
    if ($size -lt $ExpectedBytes) {
        Write-Host "Downloading $Name. Rerun setup to resume an interrupted download."
        & curl.exe --fail --location --retry 4 --continue-at - --output $path "https://downloads.carlasim.com/Windows/$Name"
        if ($LASTEXITCODE -ne 0) { throw "Download interrupted. Partial archive preserved at $path." }
    }
    if ((Get-Item -LiteralPath $path).Length -ne $ExpectedBytes) { throw "Incomplete archive at $path." }
    return $path
}

function Assert-Base([string]$Root) {
    $required = @(
        'CarlaUnreal.exe',
        'CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe',
        'CarlaUnreal/Content/Paks/pakchunk0-Windows.pak',
        'CarlaUnreal/Content/Paks/pakchunk0-Windows.ucas',
        'CarlaUnreal/Content/Paks/pakchunk0-Windows.utoc',
        'Tools/carla_pack.py',
        "PythonAPI/carla/dist/$wheelName"
    )
    foreach ($name in $required) {
        $file = Get-Item -LiteralPath (Join-Path $Root $name) -ErrorAction SilentlyContinue
        if (-not $file -or $file.Length -le 0) { throw "Missing CARLA 1.0 release file: $Root/$name" }
    }
    $stamp = (Get-Content -Raw -LiteralPath (Join-Path $Root 'BaseRelease')).Trim()
    if ($stamp -ne 'carla-1.0-Win64') { throw "Expected the CARLA 1.0 Windows release at $Root. Found $stamp." }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -Path $local -ItemType Directory -Force | Out-Null
    $python = Join-Path $PSScriptRoot '.venv-city/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (Test-Path -LiteralPath '.venv-city') { throw 'The existing .venv-city has no Python executable.' }
        $activePython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($activePython) { & $activePython.Source -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' }
        if ($activePython -and $LASTEXITCODE -eq 0) {
            & $activePython.Source -m venv .venv-city
        } elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
            & py.exe -3.12 -m venv .venv-city
        } else {
            throw 'Install 64-bit Python 3.12 for Windows, then rerun setup.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.12 environment.' }
    }
    & $python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32)'
    if ($LASTEXITCODE -ne 0) { throw 'The existing .venv-city must use 64-bit Python 3.12.' }

    $root = if ($env:CARLA_ROOT) { (Resolve-Path -LiteralPath $env:CARLA_ROOT).Path } else { Join-Path $local 'carla-1.0' }
    if ($ClientOnly) {
        & $python tools/fetch_carla_client.py $local
        if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve the CARLA 1.0 Python client.' }
        $wheel = Join-Path $local $wheelName
    } else {
        if (-not $env:CARLA_ROOT -and -not (Test-Path -LiteralPath $root)) {
            $archive = Get-ReleaseArchive $baseName 13919371161L
            $staging = Join-Path $local 'carla-1.0.staging'
            New-Item -ItemType Directory -Path $staging -Force | Out-Null
            Write-Host 'Extracting the CARLA 1.0 Windows release.'
            & tar.exe -xf $archive -C $staging --strip-components 1
            if ($LASTEXITCODE -ne 0) { throw "Extraction failed. Archive and staging files preserved at $local." }
            Assert-Base $staging
            [IO.Directory]::Move($staging, $root)
        }
        Assert-Base $root
        $wheel = Join-Path $root "PythonAPI/carla/dist/$wheelName"
        $packTool = Join-Path $root 'Tools/carla_pack.py'
        $packPath = Join-Path $root 'CarlaUnreal/Packs/AdditionalMaps'
        if (Test-Path -LiteralPath $packPath) {
            & $python $packTool verify $packPath
        } else {
            $maps = Get-ReleaseArchive $mapsName 6843655261L
            & $python $packTool install $maps --server $root
        }
        if ($LASTEXITCODE -ne 0) { throw 'The AdditionalMaps content pack failed verification or installation.' }
        $manifest = Get-Content -Raw -LiteralPath (Join-Path $packPath 'carla-pack.json') | ConvertFrom-Json
        if ($manifest.base_release -ne 'carla-1.0-Win64' -or 'Town01_Opt' -notin $manifest.maps.name) {
            throw 'The installed content pack must include Town01_Opt for this CARLA release.'
        }
    }

    & $python -m pip install --disable-pip-version-check -r requirements.txt $wheel
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the Python dependencies.' }
    & $python -c "import carla; assert carla.Client('127.0.0.1', 2000).get_client_version() == '1.0'"
    if ($LASTEXITCODE -ne 0) { throw 'CARLA client version check failed.' }
    if (-not $ClientOnly) { Write-Host "CARLA 1.0 and Town01_Opt are installed at $root." }
}
finally {
    Pop-Location
}
