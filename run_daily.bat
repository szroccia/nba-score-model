@echo off
REM Refresh this season's data, read the latest injury report and predict today's games.
REM Schedule it with Task Scheduler twice a day. See README.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m nbamodel daily --push %*
