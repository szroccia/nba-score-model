@echo off
REM Refresh this season's data, read the latest injury report and predict today's games.
REM schedule_daily.ps1 sets up Task Scheduler to run it twice a day. See README.
cd /d "%~dp0"
REM UTF-8 mode, so accented player names don't crash the run when the output goes to daily.log.
set PYTHONUTF8=1
echo ===== %date% %time% =====
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m nbamodel daily --push %*
