@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".venv-city\Scripts\python.exe" (
    echo Run setup.ps1 before opening the live console.
    popd
    exit /b 1
)
if not exist "models\fusion.npz" (
    echo The trained fusion model is missing from models\fusion.npz.
    popd
    exit /b 1
)

echo Starting the live console. Press Ctrl+C to stop the run.
".venv-city\Scripts\python.exe" -m ado drive --serve --interactive --open-browser
set "result=%errorlevel%"
popd
exit /b %result%
