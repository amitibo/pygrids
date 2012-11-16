#!/home/amitibo/epd/bin/python

from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_python_lib
from Cython.Distutils import build_ext
import numpy as np
import os.path
import sys


NAME = 'pygrids'
PACKAGE_NAME = 'grids'
VERSION = '0.1'
DESCRIPTION = 'Grid utilities'
LONG_DESCRIPTION = """
"""
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
URL = "http://bitbucket.org/amitibo/pygrids"
KEYWORDS = []
LICENSE = 'BSD'
CLASSIFIERS = [
    'License :: OSI Approved :: BSD License',
    'Development Status :: 3 - Alpha',
    'Topic :: Scientific/Engineering'
]

def main():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        license=LICENSE,
        packages=[PACKAGE_NAME],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [
            Extension(
                PACKAGE_NAME + '.' + 'cygrids',
                [
                    'src/cygrids.pyx',
                ],
                include_dirs=[np.get_include()]
            )
            ],
    )


if __name__ == '__main__':
    main()
