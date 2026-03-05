from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as f:
    readme = f.read() + "\n\n"

setup(
    name="lightcurves",
    version="0.0.2",
    include_package_data=True,
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    author="Felix Teutloff",
    author_email="astro.felix@teutloff.org",
    description="A tiny light curve visualisation and analysis tool.",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD-3-Clause",
        "Operating System :: OS Independent",
    ],
)
