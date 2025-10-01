# PowerShell script to run Python commands with the .RLtest virtual environment
# Usage: .\run_with_venv.ps1 <python_script_and_args>
# Example: .\run_with_venv.ps1 approaches/mobo/main.py

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

& ".RLtest\Scripts\python.exe" @Arguments