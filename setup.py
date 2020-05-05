import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'nexula'
AUTHOR = 'Haryo A W'
AUTHOR_EMAIL = 'haryo97@gmail.com'
URL = 'https://github.com/haryoa/nexus'
REQUIRES_PYTHON = '>=3.6.0'
LICENSE = 'MIT License'
DESCRIPTION = '(In development) Tools on Benchmarking Machine Learning Toolkit'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"


def parse_requirements():
    with open("requirements.txt", "r") as fh:
        reqs_a = fh.read()
        reqs_a = reqs_a.split('\n')
    return reqs_a


reqs = parse_requirements()


setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      python_requires=REQUIRES_PYTHON,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      entry_points={
          'console_scripts': [
              'nexula = nexula.cli:main',
          ],
      },
      install_requires=reqs,
      packages=find_packages(exclude=['tests'])
      )
