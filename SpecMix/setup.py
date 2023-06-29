from setuptools import setup, find_packages

setup(
    name='SpecMix',
    version='1.0.0',
    description='Python package for spectral clustering with mixed data types',
    authors=['Dylan Soemitro', 'Jeova Farias Sales Rocha Neto'],
    author_emails=['d.soemitro@columbia.edu', 'j.farias@bowdoin.edu'],
    url='https://github.com/dylansoemitro/SpecMix',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)