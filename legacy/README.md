# Legacy Code Archive

This folder contains archived code from the original single-approach implementation, preserved for reference.

## Contents

### Original Implementation Files
- `main.py` - Original entry point for RL-only approach
- `rl_live_demo2.py` - Alternative demo script
- `gui_pareto.py` - Standalone Pareto plotting utility
- `config.yaml` - Original RL configuration

### Original Directory Structure
- `envs/` - Original environment implementation (now in `shared/envs/`)
- `rl/` - Original RL utilities (now in `approaches/rl/`)
- `gui/` - Original GUI components (now in `shared/`)

### Planning Documents
- `mobo_planning/` - Initial MOBO design documents and outlines

### Documentation
- `VENV_USAGE.md` - Original virtual environment usage guide
- `run_with_venv.sh` - Bash wrapper script (Windows incompatible)
- `run_with_venv.bat` - CMD wrapper script

## Migration Notes

The codebase was reorganized into a modular structure:
- **Before**: Single RL approach with all code in root
- **After**: Multi-approach framework with `approaches/rl/` and `approaches/mobo/`

Key improvements in the new structure:
1. Shared components for common functionality
2. Separate configurations for each approach
3. Unified comparison framework
4. Better documentation and PowerShell support
5. Cleaner project organization

## If You Need Legacy Code

These files are preserved in case you need to:
- Reference the original implementation
- Understand the evolution of the codebase
- Extract specific functionality that wasn't migrated

For current usage, see the main README.md and documentation in `docs/`.