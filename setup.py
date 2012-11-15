#!/home/amitibo/epd/bin/python

from setuptools import setup


def main():
    """main setup function"""
    
    setup(
        name='grids',
        version='0.1',
        description='My Utilities',
        author='Amit Aides',
        author_email='amitibo@tx.technion.ac.il',
        packages=['grids'],
        #scripts=[]
        )


if __name__ == '__main__':
    main()
