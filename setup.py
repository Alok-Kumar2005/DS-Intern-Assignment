from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

def parse_requirements(fname="requirements.txt"):
    reqs = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
    return reqs

setup(
    name="DataScience-Intern",
    version="0.1.0",
    author="Alok Kumar",
    author_email="ay747283@gmail.com",
    description="DS-Intern-Assignment: a machine-learning project package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alok-Kumar2005/DS-Intern-Assignment",
    package_dir={"": "src/ml_project"},
    packages=find_packages(where="src/ml_project"),
    install_requires=parse_requirements(),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
