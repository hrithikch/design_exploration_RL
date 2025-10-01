#!/bin/bash
# Shell script to run Python commands with the .RLtest virtual environment
# Usage: ./run_with_venv.sh <python_script_and_args>
# Example: ./run_with_venv.sh approaches/mobo/main.py

.RLtest/Scripts/python.exe "$@"