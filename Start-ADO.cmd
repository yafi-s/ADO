@echo off
setlocal
pushd "%~dp0" || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\setup.ps1" -Ensure
if errorlevel 1 goto failed
call ".\Watch-Live.cmd"
set "ado_result=%errorlevel%"
popd
exit /b %ado_result%
:failed
echo Setup did not finish. Rerun this launcher to resume the download.
popd
exit /b 1
