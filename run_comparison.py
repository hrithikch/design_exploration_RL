#!/usr/bin/env python3
"""
Script to run both RL and MOBO approaches and automatically compare results.

Usage:
    python run_comparison.py --quick     # Fast comparison with reduced iterations
    python run_comparison.py --full      # Full comparison with default settings
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


def run_rl_approach(config_path: str = "approaches/rl/config.yaml", quick: bool = False):
    """Run the RL approach."""
    print("=" * 60)
    print("Running RL Approach")
    print("=" * 60)

    # Use venv Python for consistency
    python_exe = ".RLtest\\Scripts\\python.exe" if os.name == 'nt' else ".RLtest/bin/python"

    # Update config for quick run
    if quick:
        # Could modify config here for faster runs
        pass

    result = subprocess.run([python_exe, "approaches\\rl\\main.py",
                           "--config", "approaches\\rl\\config.yaml"],
                          capture_output=False, text=True)
    return result.returncode == 0


def run_mobo_approach(config_path: str = "approaches/mobo/configs/default.yaml", quick: bool = False):
    """Run the MOBO approach."""
    print("=" * 60)
    print("Running MOBO Approach")
    print("=" * 60)

    # Use venv Python for consistency
    python_exe = ".RLtest\\Scripts\\python.exe" if os.name == 'nt' else ".RLtest/bin/python"

    # Update config for quick run
    if quick:
        # Could modify config here for faster runs
        pass

    result = subprocess.run([python_exe, "approaches\\mobo\\main.py"],
                          capture_output=False, text=True)
    return result.returncode == 0


def run_comparison():
    """Run the comparison script."""
    print("=" * 60)
    print("Comparing Results")
    print("=" * 60)

    # Use venv Python for consistency
    python_exe = ".RLtest\\Scripts\\python.exe" if os.name == 'nt' else ".RLtest/bin/python"

    result = subprocess.run([python_exe, "compare.py", "--scan-latest"],
                           capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run RL vs MOBO comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Quick comparison with reduced settings")
    group.add_argument("--full", action="store_true", help="Full comparison with default settings")
    parser.add_argument("--rl-only", action="store_true", help="Run only RL approach")
    parser.add_argument("--mobo-only", action="store_true", help="Run only MOBO approach")
    parser.add_argument("--compare-only", action="store_true", help="Only run comparison on existing results")

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Starting comparison run at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    success = True

    if not args.compare_only:
        if not args.mobo_only:
            print("\n" + "="*80)
            print("STEP 1: Running RL Approach")
            print("="*80)
            if not run_rl_approach(quick=args.quick):
                print("[ERROR] RL approach failed!")
                success = False

        if not args.rl_only and success:
            print("\n" + "="*80)
            print("STEP 2: Running MOBO Approach")
            print("="*80)
            if not run_mobo_approach(quick=args.quick):
                print("[ERROR] MOBO approach failed!")
                success = False

    if success and not (args.rl_only or args.mobo_only):
        print("\n" + "="*80)
        print("STEP 3: Comparing Results")
        print("="*80)
        if not run_comparison():
            print("[ERROR] Comparison failed!")
            success = False

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    if success:
        print("[SUCCESS] Comparison run completed successfully!")
    else:
        print("[ERROR] Comparison run failed!")

    print(f"Total runtime: {duration}")
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())