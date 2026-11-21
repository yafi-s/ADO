@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".venv-city\Scripts\python.exe" (
    echo Run setup.ps1 before opening the live console.
    popd
    exit /b 1
)
if not exist "models\behavior-v2\candidate.npz" (
    echo The Town01 base model is missing from models\behavior-v2\candidate.npz.
    popd
    exit /b 1
)
if not exist "models\town01-yield-gated-v04\calibration.json" (
    echo The Town01 interaction calibration is missing from models\town01-yield-gated-v04\calibration.json.
    popd
    exit /b 1
)

echo Starting the live console. Press Ctrl+C to stop the run.
".venv-city\Scripts\python.exe" -m ado drive --policy town01-yield-gated-v04 --serve --interactive --open-browser --district heldout --scenario trip --destination market --seed 103
set "result=%errorlevel%"
popd
exit /b %result%
