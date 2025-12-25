from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(
    name='pgym',
    packages=[package for package in find_packages()
              if package.startswith('pgym')],
    install_requires=[
        'gym>=0.2.3',
        'pypower==5.1.16',
        'matplotlib',
        'networkx'
    ],
    version='0.2.0',
    description='Gym intergration for power systems',
    author='ritou11',
    url='https://github.com/ritou11/pgym',
    author_email='ritou11@gmail.com'
)
