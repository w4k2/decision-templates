from setuptools import setup, find_packages

setup(
    name="external_modules",
    version="0.1",
    package_dir={'': 'modules'},
    packages=find_packages('modules'),
)
