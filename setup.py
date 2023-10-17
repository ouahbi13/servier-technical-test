from setuptools import setup, find_packages

setup(
    name='servier',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'flask',
        'tensorflow',
        'rdkit-pypi',
        'scikeras',
    ],
    entry_points={
        'console_scripts': [
            'servier=servier.main:main',
        ],
    },
)
