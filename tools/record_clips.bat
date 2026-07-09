@echo off
rem Record raw clips from the configured cameras (offline tuning / Kaggle).
rem   record_clips.bat                          -> all cameras, 60s
rem   record_clips.bat --seconds 300            -> all cameras, 5 min
rem   record_clips.bat factory-cam-3 --seconds 120
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\record_clips.py %*
