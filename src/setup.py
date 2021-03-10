from setuptools import setup, find_packages

requirements = [
    'pandas==1.2.3',
    'shapely==1.7.1',
    'scikit-learn==0.24.1',
    'pyproj==3.0.1',
    'geojson==2.5.0'
]

setup(name='GoMapClustering',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements
)