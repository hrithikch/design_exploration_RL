# PowerShell Setup Guide

## Why PowerShell is Better for Windows

PowerShell provides several advantages over Git Bash for this project:

1. **Native Windows Integration**: Direct support for Windows paths and executables
2. **Better Virtual Environment Support**: Handles `.ps1` activation scripts properly
3. **Error Handling**: More informative error messages
4. **Performance**: Faster execution of Windows commands
5. **Consistency**: Same shell across Windows versions

## PowerShell vs Bash Changes

### Virtual Environment Activation

**Git Bash (problematic):**
```bash
# Often fails or doesn't work properly
source .RLtest/Scripts/activate
```

**PowerShell (native):**
```powershell
# Clean, native activation
.RLtest\Scripts\Activate.ps1
```

### Running Scripts

**Git Bash:**
```bash
# Need explicit .exe and forward slashes
.RLtest/Scripts/python.exe approaches/mobo/main.py
```

**PowerShell:**
```powershell
# Cleaner syntax with backslashes
.RLtest\Scripts\python.exe approaches\mobo\main.py

# Or with activation:
.RLtest\Scripts\Activate.ps1
python approaches\mobo\main.py
```

### Path Handling

**Git Bash:**
```bash
# Mixed slash handling can be inconsistent
cd approaches/mobo
../../.RLtest/Scripts/python.exe main.py
```

**PowerShell:**
```powershell
# Consistent Windows paths
cd approaches\mobo
..\..\RLtest\Scripts\python.exe main.py
```

## Setup Instructions

### 1. Enable PowerShell Script Execution

First time setup (run PowerShell as Administrator):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

Or for current user only:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Verify Setup

```powershell
# Check PowerShell version (should be 5.1+)
$PSVersionTable.PSVersion

# Check Python is available
python --version

# Test virtual environment
.RLtest\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
deactivate
```

### 3. Using the PowerShell Wrapper

```powershell
# Make script executable (if needed)
Unblock-File .\run_with_venv.ps1

# Run MOBO
.\run_with_venv.ps1 approaches\mobo\main.py

# Run RL
.\run_with_venv.ps1 approaches\rl\main.py --config approaches\rl\config.yaml

# Run comparison
.\run_with_venv.ps1 compare.py --scan-latest
```

## Recommended PowerShell Workflow

### Method 1: With Activation (Cleaner)

```powershell
# Activate virtual environment
.RLtest\Scripts\Activate.ps1

# Now use python directly
python approaches\mobo\main.py
python approaches\rl\main.py --config approaches\rl\config.yaml
python compare.py --scan-latest

# Deactivate when done
deactivate
```

### Method 2: Direct Execution (More Explicit)

```powershell
# Use full path to venv python
.RLtest\Scripts\python.exe approaches\mobo\main.py
.RLtest\Scripts\python.exe approaches\rl\main.py --config approaches\rl\config.yaml
.RLtest\Scripts\python.exe compare.py --scan-latest
```

### Method 3: Wrapper Script (Convenient)

```powershell
# Use the PowerShell wrapper
.\run_with_venv.ps1 approaches\mobo\main.py
.\run_with_venv.ps1 approaches\rl\main.py --config approaches\rl\config.yaml
.\run_with_venv.ps1 compare.py --scan-latest
```

## PowerShell-Specific Features

### Tab Completion

PowerShell provides excellent tab completion:
```powershell
# Type this and press Tab
.\run_with_venv.ps1 approaches\<TAB>
# Completes to: .\run_with_venv.ps1 approaches\mobo\
```

### Better Error Messages

PowerShell shows clearer error messages:
```powershell
# If botorch isn't installed:
.\run_with_venv.ps1 approaches\mobo\main.py
# Shows: ModuleNotFoundError: No module named 'botorch'
# With suggested solution: .\run_with_venv.ps1 -m pip install botorch
```

### Pipeline Support

PowerShell supports command chaining:
```powershell
# Run MOBO and immediately compare if successful
.\run_with_venv.ps1 approaches\mobo\main.py; if ($?) { .\run_with_venv.ps1 compare.py --scan-latest }
```

### Variables and Loops

Easy to script repetitive tasks:
```powershell
# Run multiple experiments
for ($i=1; $i -le 5; $i++) {
    Write-Host "Running experiment $i"
    .\run_with_venv.ps1 approaches\mobo\main.py
}
```

## Package Management

### Installing Packages

```powershell
# Activate environment first
.RLtest\Scripts\Activate.ps1

# Install packages
pip install botorch
pip install gpytorch
pip install -r requirements.txt

# Or without activation
.RLtest\Scripts\python.exe -m pip install botorch
```

### Checking Installation

```powershell
# List installed packages
.RLtest\Scripts\python.exe -m pip list

# Check specific package
.RLtest\Scripts\python.exe -c "import botorch; print(botorch.__version__)"
```

## Troubleshooting PowerShell Issues

### Execution Policy Error

```powershell
# Error: cannot be loaded because running scripts is disabled
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Path Not Found

```powershell
# Use Get-ChildItem to verify paths
Get-ChildItem .RLtest\Scripts\

# Use Test-Path to check existence
Test-Path .RLtest\Scripts\python.exe
```

### Module Import Errors

```powershell
# Verify you're using the right Python
.RLtest\Scripts\python.exe -c "import sys; print(sys.executable)"

# Should show: C:\path\to\your\project\.RLtest\Scripts\python.exe
```

### Performance Issues

PowerShell can be slower for some operations. For batch processing:

```powershell
# Measure execution time
Measure-Command { .\run_with_venv.ps1 approaches\mobo\main.py }
```

## IDE Integration

### VS Code

Configure VS Code to use PowerShell:

1. Open VS Code settings (Ctrl+,)
2. Search for "terminal.integrated.defaultProfile.windows"
3. Set to "PowerShell"

Configure Python interpreter:
1. Ctrl+Shift+P → "Python: Select Interpreter"
2. Choose: `.RLtest\Scripts\python.exe`

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Point to: `<project_root>\.RLtest\Scripts\python.exe`

## Migration from Bash

If you have existing bash scripts, here are the key changes:

| Bash | PowerShell |
|------|------------|
| `./script.sh` | `.\script.ps1` |
| `/path/to/file` | `\path\to\file` |
| `$?` | `$?` (same) |
| `echo` | `Write-Host` |
| `export VAR=value` | `$env:VAR = "value"` |
| `&&` | `;` (sequential) |
| `\|\|` | No direct equivalent, use if-else |

The PowerShell setup provides a more native Windows experience with better tooling integration and clearer error handling.