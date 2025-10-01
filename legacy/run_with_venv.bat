@echo off
REM Windows batch script to run Python commands with the .RLtest virtual environment
REM Usage: run_with_venv.bat <python_script_and_args>
REM Example: run_with_venv.bat approaches/mobo/main.py

.RLtest\Scripts\python.exe %*