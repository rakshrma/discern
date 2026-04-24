from setuptools import setup, find_packages
from pathlib import Path

# Read the README for the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Core dependencies required for the DISCERN evaluation pipeline
CORE_REQUIREMENTS = [
    "pandas>=1.5.0",
    "pydantic>=2.0.0",
    "PyYAML>=6.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
]

# Additional dependencies for metrics and analysis scripts
METRICS_REQUIREMENTS = [
    "numpy>=1.23.0",
    "scipy>=1.10.0",
    "matplotlib>=3.6.0",
    "seaborn>=0.12.0",
    "tqdm>=4.60.0",
    "nltk>=3.8.0",
    "rouge>=1.0.0",
    "bert-score>=0.3.13",
    "scikit-learn>=1.1.0",
    "openpyxl>=3.0.0",
]

# Radiology-specific metrics (install individually, some need checkpoints)
RADIOLOGY_REQUIREMENTS = [
    "RaTEScore",      # pip install RaTEScore
    "radgraph",       # requires PhysioNet checkpoint; see README
]

setup(
    name="discern",
    version="1.0.0",
    author="Rakesh Sharma, Cameron Beeche, Jessie Dong, Richard Zhuang, "
           "Huaizhi Qu, Ruichen Zhang, Vineeth Gangaram, Pulak Goswami, "
           "Jiayi Xin, Jenna Ballard, Ari Goldberg, Hersh Sagreiya, "
           "Qi Long, Tianlong Chen, Walter Witschey",
    description="DISCERN: A Clinical Impact-Aware Framework for Radiology Report Comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rakshrma/discern",
    packages=find_packages(exclude=["data", "data.*"]),
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "metrics":  METRICS_REQUIREMENTS,
        "radiology": RADIOLOGY_REQUIREMENTS,
        "all": CORE_REQUIREMENTS + METRICS_REQUIREMENTS + RADIOLOGY_REQUIREMENTS,
    },
    package_data={
        "config": ["*.yaml"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Academic and Industrial Radiology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "radiology",
        "report evaluation",
        "clinical NLP",
        "medical AI",
        "chest X-ray",
        "LLM evaluation",
    ],
)