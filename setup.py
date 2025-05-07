from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "DS-Intern-Assignment"
AUTHOR_USER_NAME = "Alok-Kumar2005"
SRC_REPO = "ml_project"
AUTHOR_EMAIL = "ay747283@gmail.com"

setup(
    name="imago_assignment",
    version="0.1.0",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
)