@echo off
setlocal
pushd "%~dp0" || exit /b 1
if not exist ".venv-city\Scripts\python.exe" goto install
if defined CARLA_ROOT goto drive
if exist ".local\carla-1.0\CarlaUnreal.exe" goto drive
:install
echo Installing the official CARLA 1.0 runtime, city and Python client.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\setup.ps1"
if errorlevel 1 goto failed
:drive
call ".\Watch-Live.cmd"
set "ado_result=%errorlevel%"
popd
exit /b %ado_result%
:failed
echo Setup did not finish. Rerun this launcher to resume the download.
popd
exit /b 1
