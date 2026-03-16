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
    "torch>=2.0.0",
    "transformers>=4.30.0",
]

# Additional dependencies for benchmarking and inference scripts
INFERENCE_REQUIREMENTS = [
    "numpy>=1.23.0",
    "scipy>=1.10.0",
    "matplotlib>=3.6.0",
    "tqdm>=4.60.0",
    "nltk>=3.8.0",
    "rouge>=1.0.0",
    "bert-score>=0.3.13",
]

setup(
    name="discern",
    version="0.1.0",
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
        "inference": INFERENCE_REQUIREMENTS,
        "all": CORE_REQUIREMENTS + INFERENCE_REQUIREMENTS,
    },
    package_data={
        "config": ["*.yaml"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
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
        "MICCAI",
    ],
)