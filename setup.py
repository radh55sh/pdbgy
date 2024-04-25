from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1.0',
    author='Radheesh Sharma',
    author_email='rmeda@andrew.cmu.edu',
    description='Converts a pdb file to PyG (torch_geometric)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/radh55sh/pdb2graph',
    packages=find_packages(),
    install_requires=[
        'biopython',
        'torch',
        'torch_geometric'
    ],
)
