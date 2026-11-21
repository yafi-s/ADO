param([switch]$ClientOnly, [switch]$Ensure)
$ErrorActionPreference = 'Stop'

$baseName = 'CARLA_1.0.zip'
$mapsName = 'AdditionalMaps_1.0.tar.gz'
$wheelName = 'carla-1.0-cp312-cp312-win_amd64.whl'
$local = Join-Path $PSScriptRoot '.local'
$marker = Join-Path $local 'carla-1.0.ready'
$clientProbe = "import carla, cv2, numpy, sys; sys.exit(carla.Client('127.0.0.1', 2000).get_client_version() != '1.0')"

function Get-SmallFileHash([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($Path))).Replace('-', '') }
    finally { $sha.Dispose() }
}

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
    # Byte counts from the CARLA 1.0 Windows release ZIP central directory.
    $required = @{
        'CarlaUnreal.exe' = 173568
        'CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe' = 198662656
        'CarlaUnreal/Content/Paks/global.utoc' = 806
        'CarlaUnreal/Content/Paks/global.ucas' = 3330368
        'CarlaUnreal/Content/Paks/pakchunk0-Windows.pak' = 266865663
        'CarlaUnreal/Content/Paks/pakchunk0-Windows.ucas' = 12249030576L
        'CarlaUnreal/Content/Paks/pakchunk0-Windows.utoc' = 6850115
        'CarlaUnreal/Content/Paks/pakchunk0optional-Windows.ucas' = 817664592
        'CarlaUnreal/Content/Paks/pakchunk0optional-Windows.utoc' = 230407
        'Tools/carla_pack.py' = 123010
        "PythonAPI/carla/dist/$wheelName" = 1807041
    }
    foreach ($name in $required.Keys) {
        $file = Get-Item -LiteralPath (Join-Path $Root $name) -ErrorAction SilentlyContinue
        if (-not $file -or $file.Length -ne $required[$name]) { throw "Incomplete CARLA 1.0 release file: $Root/$name" }
    }
    $stamp = (Get-Content -Raw -LiteralPath (Join-Path $Root 'BaseRelease')).Trim()
    if ($stamp -ne 'carla-1.0-Win64') { throw "Expected the CARLA 1.0 Windows release at $Root. Found $stamp." }
}

function Assert-Town([string]$Pack) {
    $manifestPath = Join-Path $Pack 'carla-pack.json'
    if ((Get-SmallFileHash $manifestPath) -ne 'EA5EDC1D79BF9D09E6D7B2E960651FF66863685B6D347B9C539BF98F4FCD7458') {
        throw 'AdditionalMaps requires the CARLA 1.0 Windows release manifest.'
    }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($manifest.base_release -ne 'carla-1.0-Win64' -or 'Town01_Opt' -notin $manifest.maps.name) {
        throw 'The installed content pack must include Town01_Opt for this CARLA release.'
    }
    foreach ($name in $manifest.files.PSObject.Properties.Name) {
        $file = Get-Item -LiteralPath (Join-Path $Pack $name) -ErrorAction SilentlyContinue
        if (-not $file -or $file.Length -le 0) { throw "Missing AdditionalMaps content: $name" }
    }
}

function Save-Incomplete([string]$Path, [string]$BackupParent, [string]$Root) {
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $active = Get-Process -Name 'CarlaUnreal*' -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) }
    if ($active) { throw "Close the running CARLA server at $Root before repairing its files." }
    $backup = Join-Path $BackupParent ((Split-Path -Leaf $Path) + '.incomplete-' + [guid]::NewGuid())
    [IO.Directory]::Move($Path, $backup)
    Write-Host "Preserved incomplete installation at $backup."
}

Push-Location -LiteralPath $PSScriptRoot
$setupLock = $null
try {
    New-Item -Path $local -ItemType Directory -Force | Out-Null
    try {
        $setupLock = [IO.File]::Open((Join-Path $local 'carla-1.0.setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] {
        throw "Could not acquire the CARLA setup lock at $local. Another setup may be running. $($_.Exception.Message)"
    }
    $python = Join-Path $PSScriptRoot '.venv-city/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
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

    if ($ClientOnly) {
        & $python tools/fetch_carla_client.py $local
        if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve the CARLA 1.0 Python client.' }
        $wheel = Join-Path $local $wheelName
    } else {
        $root = if ($env:CARLA_ROOT) { (Resolve-Path -LiteralPath $env:CARLA_ROOT).Path } else { Join-Path $local 'carla-1.0' }
        if ($env:CARLA_ROOT -and -not (Test-Path -LiteralPath (Join-Path $root 'BaseRelease'))) {
            $nested = @(Get-ChildItem -LiteralPath $root -Directory | Where-Object {
                Test-Path -LiteralPath (Join-Path $_.FullName 'BaseRelease') -PathType Leaf
            })
            if ($nested.Count -eq 1) { $root = $nested[0].FullName }
        }
        $packPath = Join-Path $root 'CarlaUnreal/Packs/AdditionalMaps'
        $receipt = "1|$root|$python|$(Get-SmallFileHash (Join-Path $PSScriptRoot 'requirements.txt'))"
        $baseReady = $false
        if (Test-Path -LiteralPath $root) {
            try { Assert-Base $root; $baseReady = $true } catch {
                if ($env:CARLA_ROOT) { throw }
                Write-Host $_.Exception.Message
            }
        }
        if ($Ensure -and $baseReady -and (Test-Path -LiteralPath $marker) -and
            [IO.File]::ReadAllText($marker) -eq $receipt) {
            try {
                Assert-Town $packPath
                & $python -c $clientProbe
                if ($LASTEXITCODE -eq 0) { return }
            } catch { Write-Host $_.Exception.Message }
        }
        # An interrupted repair must go through verification on the next launch.
        if (Test-Path -LiteralPath $marker) { [IO.File]::WriteAllText($marker, 'pending') }
        if (-not $baseReady) {
            $archive = Get-ReleaseArchive $baseName 13919371161L
            $staging = Join-Path $local 'carla-1.0.staging'
            New-Item -ItemType Directory -Path $staging -Force | Out-Null
            Write-Host 'Extracting the CARLA 1.0 Windows release.'
            & tar.exe -xf $archive -C $staging --strip-components 1
            if ($LASTEXITCODE -ne 0) { throw "Extraction failed. Archive and staging files preserved at $local." }
            Assert-Base $staging
            if (Test-Path -LiteralPath $root) { Save-Incomplete $root $local $root }
            [IO.Directory]::Move($staging, $root)
        }
        Assert-Base $root
        $wheel = Join-Path $root "PythonAPI/carla/dist/$wheelName"
        $packTool = Join-Path $root 'Tools/carla_pack.py'
        $packReady = $false
        if (Test-Path -LiteralPath $packPath) {
            & $python $packTool verify $packPath
            if ($LASTEXITCODE -eq 0) {
                try { Assert-Town $packPath; $packReady = $true } catch { Write-Host $_.Exception.Message }
            }
        }
        if (-not $packReady) {
            $maps = Get-ReleaseArchive $mapsName 6843655261L
            if (Test-Path -LiteralPath $packPath) { Save-Incomplete $packPath $root $root }
            & $python $packTool install $maps --server $root
            if ($LASTEXITCODE -ne 0) { throw 'The AdditionalMaps content pack failed verification or installation.' }
            Assert-Town $packPath
        }
    }

    & $python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('pip') is None)"
    if ($LASTEXITCODE -ne 0) {
        & $python -m ensurepip --default-pip
        if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip in the Python environment.' }
    }
    & $python -m pip install --disable-pip-version-check -r requirements.txt $wheel
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the Python dependencies.' }
    & $python -c $clientProbe
    if ($LASTEXITCODE -ne 0) { throw 'CARLA client version check failed.' }
    if (-not $ClientOnly) {
        [IO.File]::WriteAllText($marker, $receipt)
        Write-Host "CARLA 1.0 and Town01_Opt are installed at $root."
    }
}
finally {
    if ($setupLock) { $setupLock.Dispose() }
    Pop-Location
}
