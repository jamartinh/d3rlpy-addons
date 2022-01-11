from setuptools import setup

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
    install_requires=[
        "torch", "scikit-learn", "tqdm", "h5py", "gym", "d3rlpy"
    ],
)
