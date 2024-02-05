
# setup.py
from setuptools import setup
from setuptools import find_packages


with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='wine-analysis',
      #version="0.0.7" <insert version>,
      description="wine analysis for Romboost",
      license="MIT" , ## Es para poner una licencia al modelo. MIT: es para publicos,
      install_requires=requirements,
      packages=find_packages(),
      #test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
