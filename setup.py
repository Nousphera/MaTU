"""Install the source checkout and temporary native Flower application."""

from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
source = "src" if (root / "src").is_dir() else "."
requirements = [
    line.strip()
    for line in (root / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith(("#", "-e"))
]

setup(
    name="matu",
    version="0.3.0",
    description="Many-Task Federated Fine-Tuning via Unified Task Vectors",
    long_description=(
        (root / "README.md").read_text()
        if (root / "README.md").exists()
        else "MaTU research implementation."
    ),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    python_requires=">=3.11,<3.14",
    install_requires=requirements,
    package_dir={"": source},
    packages=find_packages(source, include=["matu", "matu.*"]),
    data_files=[("share/matu", ["setup.py", "requirements.txt", "LICENSE"])],
    entry_points={"console_scripts": ["matu-experiment=matu.experiment:main"]},
)
