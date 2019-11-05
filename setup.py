from setuptools import setup
from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop

import os
BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='ade',
      py_modules=['ade'],
      install_requires=[
          'dm-sonnet==1.23'
      ]
)
