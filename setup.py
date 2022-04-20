from setuptools import setup

with open("requirements.txt", "r") as req:
    requires = req.read().split("\n")


setup(name="dnosearch",
      version="0.1",
      description="Deep Neural Operator active learning with output-weighted importance sampling",
     #url="http://github.com/storborg/funniest",
      author="Ethan Pickering",
      author_email="pickering@mit.edu",
      install_requires=requires,
      packages=setuptools.find_packages(),
      include_package_data=True,
      license="MIT"
    )
