"""
PicoTuri EditJudge Setup Script
Privacy-first edit quality assessment system
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version information
VERSION = "0.2.0"

setup(
    name="picoturi-editjudge",
    version=VERSION,
    author="PicoTuri Team",
    author_email="team@picoturi.ai",
    description="Privacy-first edit quality assessment system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/picoturi/editjudge",
    project_urls={
        "Bug Tracker": "https://github.com/picoturi/editjudge/issues",
        "Documentation": "https://picoturi-editjudge.readthedocs.io/",
        "Source Code": "https://github.com/picoturi/editjudge",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
            "isort>=5.12.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.1.0",
        ],
        "web": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "streamlit>=1.28.0",
        ],
        "mobile": [
            "coremltools>=7.0.0",
            "tensorflow-lite>=2.13.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "sphinx>=7.1.0",
            "fastapi>=0.104.0",
            "coremltools>=7.0.0",
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "editjudge-train=src.train.fusion_trainer:main",
            "editjudge-export=src.export.onnx_export:main",
            "editjudge-serve=src.serving.api:main",
            "editjudge-evaluate=src.evaluation.evaluator:main",
            "editjudge-calibrate=src.adaptation.calibration:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.txt",
            "configs/**/*.yaml",
            "configs/**/*.yml",
            "configs/**/*.json",
            "models/**/*.onnx",
            "models/**/*.pt",
            "models/**/*.pth",
            "assets/**/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "computer vision",
        "natural language processing",
        "edit quality assessment",
        "multimodal learning",
        "onnx",
        "privacy",
        "cross-platform",
    ],
)
