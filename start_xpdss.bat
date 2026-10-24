@echo off
cd /d "%~dp0"
start "" /b pythonw.exe audio_analyzer.py %*
