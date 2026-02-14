import os
import subprocess
from datetime import datetime
from pathlib import Path

import click
import yaml

# Load configuration
CONFIG_FILE = "project_config.yaml"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


config = load_config()
DIRS = config.get("directories", {})
TEST_CONFIG = config.get("testing", {})


@click.group()
def cli():
    """Project management CLI."""
    pass


@cli.command()
def test():
    """Run unit tests and generate timestamped reports."""
    test_dir = DIRS.get("tests", "tests")
    results_dir = DIRS.get("test_results", "unit test results")

    # Create a timestamped report file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = f"results_{timestamp}.xml"

    # Ensure directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    report_path = os.path.join(results_dir, report_file)
    cmd = ["pytest", test_dir, f"--junitxml={report_path}"]

    # Add project root to PYTHONPATH to allow src imports
    env = os.environ.copy()
    project_root = str(Path(__file__).parent)
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    click.echo(f"Running tests in {test_dir} and saving report to {report_path}...")
    subprocess.run(cmd, env=env, check=False)


@cli.command()
def lint():
    """Run linting checks."""
    src_dir = DIRS.get("src", "src")
    cmd = ["ruff", "check", src_dir]
    click.echo(f"Linting {src_dir}...")
    subprocess.run(cmd, check=False)


@cli.command()
def run():
    """Run the application."""
    src_dir = DIRS.get("src", "src")
    main_file = os.path.join(src_dir, "main.py")
    cmd = ["python", main_file]
    click.echo(f"Running {main_file}...")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    cli()
