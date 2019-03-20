"""
TRIPOL related package
"""

from setuptools import setup, find_packages

setup_requires=[]
install_requires = ['numpy',
                    'scipy',
                    'astropy >= 2.0',
                    'ccdproc >= 1.3']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3.6"]

setup(
    name="tripolpy",
    version="0.1.2",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="Data reduction package for TRIPOL at SNU",
    license="MIT",
    keywords="",
    url="https://github.com/ysBach/TRIPOLpy",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires )
