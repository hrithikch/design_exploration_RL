# Virtual Environment Usage Guide

## The Virtual Environment Issues Explained

The virtual environment problems occurred due to differences between Windows activation methods and Git Bash compatibility:

### Why Activation Failed in Git Bash

1. **Windows vs Unix paths**: Windows uses `Scripts\` while Unix uses `bin/`
2. **Batch file compatibility**: Git Bash doesn't handle `.bat` files well
3. **PowerShell scripts**: `.ps1` files require execution policy changes

## Proper Virtual Environment Usage

### Method 1: Use Direct Python Path (Recommended for Git Bash)
```bash
# From project root
.RLtest/Scripts/python.exe <your_script.py>

# Examples:
.RLtest/Scripts/python.exe approaches/mobo/main.py
.RLtest/Scripts/python.exe approaches/rl/main.py
.RLtest/Scripts/python.exe compare.py --scan-latest
```

### Method 2: Use Wrapper Scripts (Easiest)
```bash
# Use the provided wrapper scripts
./run_with_venv.sh approaches/mobo/main.py
./run_with_venv.sh approaches/rl/main.py
./run_with_venv.sh compare.py --scan-latest
```

### Method 3: Proper Activation (Shell-specific)

**Windows Command Prompt:**
```cmd
.RLtest\Scripts\activate.bat
python approaches/mobo/main.py
deactivate
```

**Windows PowerShell:**
```powershell
.RLtest\Scripts\Activate.ps1
python approaches/mobo/main.py
deactivate
```

**Git Bash (Linux/Mac style):**
```bash
# Activation often doesn't work, use direct path instead
source .RLtest/Scripts/activate  # May not work
.RLtest/Scripts/python.exe approaches/mobo/main.py  # Use this instead
```

## Best Practices

### 1. Check Your Environment
```bash
# Verify which Python you're using
which python
python --version

# With venv (should show venv path)
.RLtest/Scripts/python.exe --version
```

### 2. Install Packages Correctly
```bash
# Always use the venv Python for package installation
.RLtest/Scripts/python.exe -m pip install <package>

# NOT: pip install <package>  # This might install to system Python
```

### 3. IDE Configuration
If using an IDE, point it to:
- **Python Interpreter**: `<project_root>/.RLtest/Scripts/python.exe`
- **Pip Path**: `<project_root>/.RLtest/Scripts/pip.exe`

## Quick Commands Reference

```bash
# Install new package
.RLtest/Scripts/python.exe -m pip install <package_name>

# List installed packages
.RLtest/Scripts/python.exe -m pip list

# Run MOBO with GUI
.RLtest/Scripts/python.exe approaches/mobo/main.py

# Run RL approach
.RLtest/Scripts/python.exe approaches/rl/main.py

# Run comparison
.RLtest/Scripts/python.exe compare.py --scan-latest

# Using wrapper script (after chmod +x run_with_venv.sh)
./run_with_venv.sh approaches/mobo/main.py
```

## Troubleshooting

### "No module named 'botorch'" Error
```bash
# Install missing packages with venv Python
.RLtest/Scripts/python.exe -m pip install botorch gpytorch

# Verify installation
.RLtest/Scripts/python.exe -c "import botorch; print('BoTorch installed successfully')"
```

### "Permission Denied" on Scripts
```bash
# Make wrapper script executable
chmod +x run_with_venv.sh

# Or use direct path
.RLtest/Scripts/python.exe approaches/mobo/main.py
```

### Wrong Python Being Used
```bash
# Check current Python
which python

# Use explicit venv Python path
.RLtest/Scripts/python.exe <your_script>
```

## Why This Approach is Better

1. **Explicit**: Always clear which Python you're using
2. **Portable**: Works across different shells and OS
3. **Reliable**: Doesn't depend on environment activation
4. **Debuggable**: Easy to verify you're using the right environment

The wrapper scripts provide the convenience of short commands while maintaining the reliability of explicit paths.