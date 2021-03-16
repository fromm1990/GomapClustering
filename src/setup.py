import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

requirements = [
    'pandas==1.2.3',
    'shapely==1.7.1',
    'scikit-learn==0.24.1',
    'pyproj==3.0.1',
    'geojson==2.5.0',
    'optuna==2.6.0',
    'wandb==0.10.22',
    'plotly==4.14.3',
    'Cython==0.29.22',
    'geopandas==0.9.0',
    'contextily==1.1.0'
]

setup(
    name='GoMapClustering',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=cythonize('GoMapClustering/AngularClustering/fcm4dd.pyx'),
    include_dirs=[numpy.get_include()]
)
