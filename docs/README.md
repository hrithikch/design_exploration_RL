# Documentation Index

## Quick Links

- **[User Guide](user-guide.md)** - How to run experiments and configure settings
- **[Developer Guide](developer-guide.md)** - Technical details and extending the framework
- **[PowerShell Setup](powershell-setup.md)** - Windows-native setup instructions

## Documentation Overview

### For Users

Start with the **User Guide** if you want to:
- Run PPA optimization experiments
- Compare RL vs MOBO approaches
- Understand configuration options
- Interpret results and plots

### For Developers

Read the **Developer Guide** if you want to:
- Understand the codebase architecture
- Add new optimization methods
- Integrate with real EDA tools
- Extend to new problem domains

### For Windows Users

Check the **PowerShell Setup** guide for:
- Native Windows development experience
- Better virtual environment handling
- Improved error messages and debugging
- IDE integration tips

## Getting Started

1. **Installation**: Follow setup instructions in [User Guide](user-guide.md#installation)
2. **Quick Test**: Run `.\run_with_venv.ps1 approaches\mobo\main.py`
3. **Comparison**: Run `.\run_with_venv.ps1 compare.py --scan-latest`
4. **Configuration**: Modify configs as described in [User Guide](user-guide.md#configuration-guide)

## Project Structure Reference

```
RL/
├── docs/                    # Documentation (you are here)
│   ├── user-guide.md       # Running experiments, configuration
│   ├── developer-guide.md  # Technical details, extensibility
│   └── powershell-setup.md # Windows-specific setup
├── approaches/             # Optimization methods
│   ├── rl/                # Reinforcement Learning
│   └── mobo/              # Multi-Objective Bayesian Optimization
├── shared/                # Common components
│   ├── envs/              # Problem environments
│   └── utils/             # Shared utilities
├── runs/                  # Experiment results
└── comparison_plots/      # Comparison visualizations
```

## Troubleshooting Quick Reference

| Issue | Solution | Guide |
|-------|----------|-------|
| Module not found | `.\run_with_venv.ps1 -m pip install <module>` | [PowerShell Setup](powershell-setup.md) |
| Execution policy error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` | [PowerShell Setup](powershell-setup.md) |
| Poor results | Adjust configuration parameters | [User Guide](user-guide.md#configuration-guide) |
| GUI doesn't show | Check Tkinter: `python -c "import tkinter"` | [User Guide](user-guide.md#troubleshooting) |
| Want to add new method | Follow modular design patterns | [Developer Guide](developer-guide.md#modular-design--extensibility) |

## Contribution Guidelines

When adding to the documentation:

1. **User-focused content** goes in `user-guide.md`
2. **Technical/implementation details** go in `developer-guide.md`
3. **Windows/PowerShell specific** content goes in `powershell-setup.md`
4. **Cross-references** between documents for related topics
5. **Code examples** should be tested and working
6. **Screenshots** for GUI features when helpful

## External Resources

- [BoTorch Documentation](https://botorch.org/) - Multi-objective Bayesian optimization
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Reinforcement learning
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [PowerShell Documentation](https://docs.microsoft.com/en-us/powershell/) - Windows scripting