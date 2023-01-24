"""Setup file for pypi package called koopa-luigi."""

# python setup.py sdist
# twine upload dist/latest-version.tar.gz

from setuptools import find_packages
from setuptools import setup

setup(
    # Description
    name="koopa-luigi",
    version="0.0.1",
    license="MIT",
    description="A functional implementation of koopa",
    # Installation
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["koopa", "luigi"],
    entry_points={"console_scripts": ["koopa-luigi = src.cli:main"]},
    # Metadata
    author="Bastian Eichenberger",
    author_email="bastian@eichenbergers.ch",
    url="https://github.com/bbquercus/koopa/",
    project_urls={
        "Documentation": "https://github.com/BBQuercus/koopa-luigi/README.md",
        "Changelog": "https://github.com/BBQuercus/koopa-luigi/releases",
        "Issue Tracker": "https://github.com/bbquercus/koopa-luigi/issues",
    },
    keywords=["biomedical", "bioinformatics", "image analysis"],
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Utilities",
    ],
)
