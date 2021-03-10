import json
import math
import os
from typing import Any, Callable, Dict, Iterable, List, TypeVar

import geojson
import numpy as np
from numpy.lib.function_base import select
import pyproj
from geojson import Feature, FeatureCollection
from shapely.geometry import Point
from shapely.ops import cascaded_union, transform

PI = math.pi
TWO_PI = 2 * math.pi

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')


class AngleIndexer:
    """Group angeles into slices in the unit circle.
    """

    def __init__(self, size: int=36) -> None:
        """Group angeles into slices in the unit circle.

        Parameters
        ----------
        size : int, optional
            The amount of slices the unit circle should be divided into, by default 36
        """
        self.slice = TWO_PI / size
    
    def index(self, angle: float) -> int:
        return math.floor(((angle + self.slice / 2) / self.slice) % (TWO_PI / self.slice))


def to_compass_angle(angle: float) -> float:
    '''
    Converts angles from [-PI, PI] to [0, 2*PI)
    # Paramters
    - angle (float): The angle to convert
    '''
    return (angle + TWO_PI) % TWO_PI


def group_by(
    elements: Iterable[TValue], 
    key_predicate: Callable[[TValue], TKey], 
    value_predicate: Callable[[TValue], Any] = None
) -> Dict[TKey, List[TValue]]:
    result = {}
    for element in elements:
        key = key_predicate(element)

        if key not in result:
            result[key] = []

        if value_predicate is None:
            result[key].append(element)
        else:
            result[key].append(value_predicate(element))

    return result


def get_file_dir(filepath: str) -> str:
    """
    Returns the absolute directory path containing the given file
    """
    return os.path.dirname(os.path.realpath(filepath))





def f1_ij(truth, prediction, i, j):
    j_count = np.int32(0)
    i_count = np.int32(0)
    ji_overlap = np.int32(0)

    for k in range(0, len(truth)):
        if prediction[k] == j:
            j_count = j_count + 1

            if truth[k] == i:
                ji_overlap = ji_overlap + 1

        if truth[k] == i:
            i_count = i_count + 1

    precision = ji_overlap / j_count
    recall = ji_overlap / i_count

    return 2 * precision * recall / (precision + recall)


def f1_score(truth, prediction, weighted=False):
    # The first cluster ID is 0, hence + 1
    true_cluster_count = max(truth) + 1
    # The first cluster ID is 0, hence + 1
    predicted_cluster_count = max(prediction) + 1
    max_values = []

    for i in range(true_cluster_count):
        i_max = 0

        for j in range(predicted_cluster_count):
            result = f1_ij(truth, prediction, i, j)
            if result > i_max:
                i_max = result

        max_values.append(i_max)

    if not weighted:
        return sum(max_values) / true_cluster_count

    # Calculate the weighted f1_score
    rv = 0
    for i in range(true_cluster_count):
        points_is_cluster_i = sum(map(lambda x: x == i, truth))
        rv = rv + points_is_cluster_i / len(truth) * max_values[i]

    return rv


def compute_cluster_centroid(cluster: List[Point]) -> Point:
    return cascaded_union(cluster).centroid
