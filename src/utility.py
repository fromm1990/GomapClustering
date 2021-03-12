import json
import math
from typing import Callable, Iterable, List

import geojson
import pyproj
from geojson import Feature, FeatureCollection, Point
from pandas import DataFrame, Series
from shapely.ops import cascaded_union, transform

from GoMapClustering.base import GoMapClusterMixin
from model import Position

PI = math.pi
TWO_PI = 2 * math.pi


class AngleIndexer:
    """Group angeles into slices in the unit circle.
    """

    def __init__(self, size: int = 36) -> None:
        """Group angeles into slices in the unit circle.

        Parameters
        ----------
        size : int, optional
            The amount of slices the unit circle should be divided into, by default 36
        """
        self.slice = TWO_PI / size

    def index(self, angle: float) -> int:
        return math.floor(((angle + self.slice / 2) / self.slice) % (TWO_PI / self.slice))


def compute_cluster_centroid(cluster: List[Point]) -> Point:
    return cascaded_union(cluster).centroid


def signed_angle_diff(source: float, target: float) -> float:
    """Calculates the shortest angle difference in radians.
    The angle differnece is signed such signed_angle_diff(1/2*PI, 0) => -1/2*PI.

    Parameters
    ----------
    source : float
        A radian angle
    target : float
        A radian angle

    Returns
    -------
    float
        Returns the minimum angle difference in radians
    """
    return ((target - source + PI) % TWO_PI) - PI


def load_geoJson(filepath: str, sourceCrs: str, targetCrs: str) -> Iterable[Position]:
    source = pyproj.CRS(sourceCrs)
    target = pyproj.CRS(targetCrs)
    project = pyproj.Transformer.from_crs(source, target, always_xy=True)

    with open(filepath, 'r') as file:
        features = json.load(file)['features']

        for feature in features:
            if feature['geometry'] is None:
                continue

            geom = Position(feature['geometry']['coordinates'])
            geom = transform(project.transform, geom)
            geom.properties = feature['properties']

            yield geom


def dump_geoJson(filepath: str, sourceCrs: str, targetCrs: str, points: Iterable[Position]):
    source = pyproj.CRS(sourceCrs)
    target = pyproj.CRS(targetCrs)
    project = pyproj.Transformer.from_crs(source, target, always_xy=True)

    features = []
    for point in points:
        transformed_point = transform(project.transform, point)
        feature_point = Point(
            [transformed_point.x, transformed_point.y], precision=15)
        feature = Feature(None, feature_point, point.properties)
        features.append(feature)

    featureCollection = FeatureCollection(features)

    with open(filepath, 'w') as file:
        file.write(geojson.dumps(featureCollection))


def cluster(
    df: DataFrame,
    cluster_algo: GoMapClusterMixin
) -> DataFrame:
    rv = DataFrame()
    for _, group in df.groupby('classifier'):
        group['cid'] = cluster_algo.cluster(group)
        rv = rv.append(group)

    return rv


def aggregate(df: DataFrame, cluster_singularity: Callable[[List[Position]], Position]) -> DataFrame:
    size = 36
    indexer = AngleIndexer(size)
    df['angle_index'] = [indexer.index(x) for x in df['heading']]
    df['index_heading'] = [x * TWO_PI / size for x in df['angle_index']]

    return df\
        .groupby(['cid', 'classifier'], as_index=False)\
        .agg(
            geom=('geom', cluster_singularity),
            avg_score=('score', 'mean'),
            avg_speed=('speed', 'mean'),
            avg_heading=('heading', 'mean'),
            heading=('index_heading', lambda x: Series.mode(x)[0]),
            count=('geom', 'size'),
            trip_count=('trip_id', 'nunique')
        ).set_index(['cid', 'classifier'])


def get_predictions(
    df: DataFrame,
    cluster_algo: GoMapClusterMixin,
    cluster_singularity: Callable[[List[Position]], Position]
) -> DataFrame:
    # Cluster the data
    df = cluster(df, cluster_algo)
    # Remove outliers
    df = df[df['cid'] > -1]
    # Aggregate the data
    return aggregate(df, cluster_singularity)
