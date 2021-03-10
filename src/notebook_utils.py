import math
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from GoMapClustering import utility
from GoMapClustering.base import GoMapClusterMixin
from GoMapClustering.utility import TWO_PI, AngleIndexer
from model import Position
from numpy import ndarray
from pandas import DataFrame, Series
from shapely.geometry.point import Point
from shapely.strtree import STRtree


class Prediction(Position):
    def __init__(self, prediction: Position, truth: Position = None):
        super().__init__(prediction)
        self.truth = truth


# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
# https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def ema_smoothing(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        if math.isnan(point):
            smoothed.append(smoothed[-1])
            continue
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def mse(predictions: ndarray, truths: ndarray, metric: Callable[[float, float], float] = None) -> float:
    if len(predictions) != len(truths):
        raise ValueError('There must be exactly one truth for each prediction')
    # If both arryes has the same length it's enough to check if one of them is empty
    if len(predictions) == 0:
        return float('NaN')
    if not isinstance(predictions, ndarray):
        predictions = np.array(predictions)
    if not isinstance(truths, ndarray):
        truths = np.array(truths)

    if metric is None:
        return ((predictions - truths)**2).mean(axis=None)

    data = np.column_stack((predictions, truths))
    return np.array([metric(x[0], x[1])**2 for x in data]).mean(axis=None)


def rmse(predictions: ndarray, truths: ndarray, metric: Callable[[float, float], float] = None) -> float:
    return math.sqrt(mse(predictions, truths, metric))


def compute_angle_score(angle_1: float, angle_2: float) -> float:
    """Computes a score in the range [0, 1].
    0 is returned when the no angle difference is messured.
    1 is returned when the maximum angle difference is messured.

    Parameters
    ----------
    angle_1 : float
        A radian angle.
    angle_2 : float
        A radian angle.

    Returns
    -------
    float
        Returns a score between 0 and 1. 
    """
    signed_angle_diff = utility.signed_angle_diff(angle_1, angle_2)
    return 1 - abs(signed_angle_diff) / utility.PI


def compute_distance_score(dist: float, max_dist: float) -> float:
    return 1 - dist / max_dist


def compute_score(cluster_center: Point, heading: float, truth: Position, max_dist: float) -> float:
    angle_score = compute_angle_score(truth.properties['heading'], heading)
    distance_score = compute_distance_score(truth.distance(cluster_center), max_dist)
    return (angle_score + distance_score) / 2


def compute_scores(cluster_center: Point, heading: float, truths: List[Position]) -> Tuple[float, Position]:
    max_distance = max((x.distance(cluster_center) for x in truths))

    for truth in truths:
        yield (compute_score(cluster_center, heading, truth, max_distance), truth)


def find_truths(
    classifier: str,
    cluster_center: Point,
    heading: float,
    truths: STRtree
) -> Union[None, Position]:
    query = cluster_center.buffer(20)
    results = (x for x in truths.query(query) if x.intersects(query))
    # The prediction type should be the same as the truth
    results = [x for x in results if x.properties['label_name'] == classifier]

    if len(results) == 0:
        return []

    # Assign a score to each candidate
    results = compute_scores(cluster_center, heading, results)
    # Return ordered results
    return (x[1] for x in sorted(results, key=lambda x: x[0], reverse=True))


def compute_precision(true_positives: int, false_positives: int) -> float:
    denominator = true_positives + false_positives
    return float('NaN') if denominator == 0 else true_positives / denominator


def compute_recall(true_positives: int, false_negatives: int) -> float:
    denominator = true_positives + false_negatives
    return float('NaN') if denominator == 0 else true_positives / denominator


def compute_f1(precision: float, recall: float) -> float:
    if float('NaN') in {precision, recall}:
        return float('NaN')

    denominator = (precision + recall)

    if denominator == 0:
        return float('NaN')

    return 2 * (precision * recall / denominator)


def compute_metrics(predictions: DataFrame, truths: List[Point]) -> Dict[str, float]:
    def get_truth_heading(truth: Point):
        return truth.properties['heading']
    def get_xy(point: Point):
        return Series(np.array([point.x, point.y]))

    test = predictions[predictions['truth'].notnull()]

    true_positives = len(test)
    false_positives = len(predictions) - true_positives
    false_negatives = len(truths) - true_positives

    y_true = test['truth'].apply(get_xy)
    y_pred = test['geom'].apply(get_xy)
    angle_true = test['truth'].apply(get_truth_heading)
    angle_pred = test['heading']

    precision = compute_precision(true_positives, false_positives)
    recall = compute_recall(true_positives, false_negatives)
    f1 = compute_f1(precision, recall)

    return {
        'rmse_location': rmse(y_pred, y_true),
        'rmse_angle': math.degrees(rmse(angle_pred, angle_true, utility.signed_angle_diff)),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def assign_truths(predictions: DataFrame, truths: STRtree):
    visited = set()
    truth_column = []

    for classifier, cluster_center, heading in zip(
        predictions.index.get_level_values('classifier'),
        predictions['geom'],
        predictions['heading']
    ):
        truth = None
        for candidate in find_truths(classifier, cluster_center, heading, truths):
            if candidate not in visited:
                truth = candidate
                visited.add(candidate)
                break
        truth_column.append(truth)

    predictions['truth'] = truth_column


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
