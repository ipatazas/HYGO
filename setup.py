from setuptools import setup, find_packages

setup(
    name='HYGO',
    version='0.9.8',
    description='HYGO: A genetic optimization framework',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ipatazas/HYGO",
    author='Isaac Robledo Martín',
    packages=find_packages(),  
    include_package_data=True,
    package_data={
        'hygo.tools': ['Parameter_help.txt'],  
    },
    install_requires=[
        'numpy>=1.16,<2.0',
        'scipy>=1.3',
        'matplotlib>=3.0',
        'pandas>=0.25',
        'dill>=0.3.0',
        'scipy>=1.15',
    ],
    python_requires='>=3.7'
)
