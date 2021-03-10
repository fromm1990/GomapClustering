import json
import math
from typing import Iterable

import geojson
import pyproj
from geojson import Feature, FeatureCollection, Point
from shapely.ops import transform

from model import Position

PI = math.pi
TWO_PI = 2 * math.pi

def signed_angle_diff(source: float, target: float) -> float:
    """Calculates the shortest angle difference in radians.
    The angle differnece is signed such signed_angle_diff(90, 0) => -90.

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
