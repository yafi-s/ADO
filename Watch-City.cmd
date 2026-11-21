@echo off
setlocal
pushd "%~dp0" || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\setup.ps1" -Ensure
if errorlevel 1 goto failed
".venv-city\Scripts\python.exe" -m ado city-drive --serve --open-browser %*
set "ado_result=%errorlevel%"
popd
exit /b %ado_result%
:failed
echo Setup did not finish. Run this launcher again to resume the download.
popd
exit /b 1
