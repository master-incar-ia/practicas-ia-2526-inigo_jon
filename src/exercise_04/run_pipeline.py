"""
Sequential pipeline runner for exercise_02: dataset -> model -> train -> evaluate
"""

import subprocess
import sys


def run_module(module_name: str) -> int:
    """Run a Python module and return exit code."""
    print(f"\n{'=' * 60}")
    print(f"Running: {module_name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=".",
    )
    return result.returncode


def main():
    """Execute pipeline modules sequentially."""
    modules = [
        "src.exercise_04.dataset",
        "src.exercise_04.model",
        "src.exercise_04.train",
        "src.exercise_04.evaluate",
    ]

    for module in modules:
        exit_code = run_module(module)
        if exit_code != 0:
            print(f"\n❌ Error: {module} failed with exit code {exit_code}")
            return exit_code

    print(f"\n{'=' * 60}")
    print("✅ Full pipeline completed successfully!")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
