from setuptools import setup

# I guess that if you are installing a library d3rlpy-addons you have installed and are using d3rlpy
# so I will drop any requirements from this install setup
setup(
    name='d3rlpy-addons',
    version='0.1',
    packages=[
        'd3rlpy_addons',
        'd3rlpy_addons.fitters',
        "d3rlpy_addons.models"
    ],
    url='',
    license='MIT',
    author='Jose Antonio Martin H.',
    author_email='xjamartinh@gmail.com',
    description='Addons for d3rpy RL library',
    install_requires=[],
)
