from setuptools import setup, find_packages

setup(
    name="discern",
    version="0.1.0",
    description="DISCERN: LLM-based clinical evaluation of radiology reports",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "nltk>=3.8.0",
        "rouge>=1.0.0",
        "bert-score>=0.3.13",
    ],
)
