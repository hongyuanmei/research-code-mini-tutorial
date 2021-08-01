import setuptools
from setuptools import setup

setup(
    name='simpclass',
    version='1.0',
    description='a simple classification task',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch==1.9.0',
        'numpy', 
        'matplotlib'
        ]
    )