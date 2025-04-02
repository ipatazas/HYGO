from setuptools import setup, find_packages

setup(
    name='HYGO',
    version='0.1.0',
    description='HYGO: A genetic optimization framework',
    author='Isaac Robledo Martín',
    packages=find_packages(),  # Automatically finds all packages (HYGO, HYGO_tools, Plant, etc.)
    install_requires=[
        'numpy>=1.16',
        'scipy>=1.3',
        'matplotlib>=3.0',
        'pandas>=0.25',
        'dill>=0.3.0'
    ],
    include_package_data=True,
    python_requires='>=3.7'
)
