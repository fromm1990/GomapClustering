import math
from typing import List, Tuple, Union

import numpy as np
from geopandas import GeoDataFrame
from numpy import ndarray
from pandas import DataFrame
from pandas.core.series import Series
from shapely.geometry import Point
from shapely.strtree import STRtree

import utility
from model import Position


class GoMapEvaluator:
    def __init__(self, predictions: DataFrame, truths: STRtree):
        # Store a reference to the data the evaluator is based on.
        self.truths = truths
        self.predictions = predictions

        # Store the data table containing the mapping between truths an predictions
        predictions = GeoDataFrame(
            predictions[['geom', 'heading']],
            crs='epsg:3044',
            geometry='geom'
        )
        self.data: GeoDataFrame = self.__get_truths(predictions)\
            .set_index(predictions.index)\
            .join(predictions)\
            .dropna()

    class Evaluation:
        errors: DataFrame
        __true_positives: float
        __false_positives: float
        __false_negatives: float
        __precision: float = None
        __recall: float = None
        __f1: float = None
        __mae_direction: float = None
        __mse_direction: float = None
        __rmse_direction: float = None
        __mae_location: float = None
        __mse_location: float = None
        __rmse_location: float = None

        def __init__(self, predictions, truths, errors) -> None:
            self.errors = errors
            self.__true_positives = len(errors)
            self.__false_positives = len(predictions) - self.__true_positives
            self.__false_negatives = len(truths._geoms) - self.__true_positives

        # Direction
        @property
        def mae_direction(self) -> float:
            if self.__mae_direction is None:
                self.__mae_direction = np.mean(
                    self.errors['direction_error']
                )
            return self.__mae_direction

        @property
        def mae_direction_degrees(self) -> float:
            return math.degrees(self.mae_direction)

        @property
        def mse_direction(self) -> float:
            if self.__mse_direction is None:
                self.__mse_direction = np.mean(
                    self.errors['direction_error']**2
                )
            return self.__mse_direction

        @property
        def mse_direction_degrees(self) -> float:
            return math.degrees(self.mse_direction)

        @property
        def rmse_direction(self) -> float:
            if self.__rmse_direction is None:
                self.__rmse_direction = math.sqrt(self.mse_direction)
            return self.__rmse_direction

        @property
        def rmse_direction_degrees(self) -> float:
            return math.degrees(self.rmse_direction)

        # Location
        @property
        def mae_location(self) -> float:
            if self.__mae_location is None:
                self.__mae_location = np.mean(self.errors['location_error'])
            return self.__mae_location

        @property
        def mse_location(self) -> float:
            if self.__mse_location is None:
                self.__mse_location = np.mean(self.errors['location_error']**2)
            return self.__mse_location

        @property
        def rmse_location(self) -> float:
            if self.__rmse_location is None:
                self.__rmse_location = math.sqrt(self.mse_location)
            return self.__rmse_location

        @property
        def precision(self):
            if self.__precision is not None:
                return self.__precision

            denominator = self.__true_positives + self.__false_positives
            self.__precision = math.nan if denominator == 0 else self.__true_positives / denominator
            return self.__precision

        @property
        def recall(self):
            if self.__recall is not None:
                return self.__recall

            denominator = self.__true_positives + self.__false_negatives
            self.__recall = math.nan if denominator == 0 else self.__true_positives / denominator
            return self.__recall

        @property
        def f1(self):
            if self.__f1 is not None:
                return self.__f1

            denominator = self.precision + self.recall
            if math.isnan(denominator) or denominator == 0:
                self.__f1 = math.nan
            else:
                self.__f1 = 2 * (self.precision * self.recall / denominator)

            return self.__f1

    def evaluate(self) -> Evaluation:
        errors = {
            'direction_error': self.__get_direction_errors(),
            'location_error': self.__get_location_errors()
        }
        errors = DataFrame(errors, index=errors['direction_error'].index)
        return self.Evaluation(self.predictions,  self.truths, errors)

    def __get_direction_errors(self) -> Series:
        return self.__get_angle_error(
            self.data['heading'],
            self.data['true_heading']
        )

    def __get_location_errors(self) -> Series:
        return self.data['geom'].distance(self.data['true_geom'])

    @staticmethod
    def __get_angle_error(
        prediction: Union[Series, DataFrame, ndarray, float],
        truth: Union[Series, DataFrame, ndarray, float]
    ) -> Union[Series, float]:
        supported_types = (Series, DataFrame, ndarray, float)
        if not isinstance(prediction, supported_types):
            raise TypeError(
                f'Expected types for param "prediction" [Series, DataFrame, ndarray or float], got {type(prediction)}')
        if not isinstance(truth, supported_types):
            raise TypeError(
                f'Expected types for param "truth" [Series, DataFrame, ndarray or float], got {type(prediction)}')

        return abs(utility.signed_angle_diff(prediction, truth))

    @staticmethod
    def __get_location_error(prediction: Point, truth: Point) -> Union[float, None]:
        if prediction is None or truth is None:
            return None

        return prediction.distance(truth)

    def __get_candidate_errors(self, cluster_center: Point, heading: float, candidates: List[Position]) -> Tuple[float, Position]:
        norm_location_errors = (self.__get_location_error(
            cluster_center, x) / 20 for x in candidates)
        norm_angle_errors = (self.__get_angle_error(
            heading, x.properties['heading']) / 180 for x in candidates)
        norm_errors = (
            x + y for x, y in zip(norm_location_errors, norm_angle_errors))
        return zip(norm_errors, candidates)

    def __get_truth_candidates(
        self,
        classifier: str,
        cluster_center: Point,
        heading: float
    ) -> Union[None, Position]:
        query = cluster_center.buffer(20)
        results = (x for x in self.truths.query(query) if x.intersects(query))
        # The prediction type should be the same as the truth
        results = [x for x in results if x.properties['label_name'] == classifier]

        if len(results) == 0:
            return []

        # Assign a score to each candidate
        results = self.__get_candidate_errors(cluster_center, heading, results)
        # Return ordered results
        return (x[1] for x in sorted(results, key=lambda x: x[0]))

    def __get_truths(self, df: DataFrame) -> GeoDataFrame:
        data = zip(
            df.index.get_level_values('classifier'),
            df['geom'],
            df['heading']
        )

        visited = set()

        def get_truth(classifier: str, cluster_center: Point, heading: float) -> Union[Tuple[Point, float], None]:
            for candidate in self.__get_truth_candidates(classifier, cluster_center, heading):
                if candidate not in visited:
                    visited.add(candidate)
                    return (candidate, candidate.properties['heading'])
            return (None, None)

        return GeoDataFrame(
            [
                get_truth(classifier, cluster_center, heading)
                for classifier, cluster_center, heading
                in data
            ],
            columns=['true_geom', 'true_heading'],
            crs='epsg:3044',
            geometry='true_geom',
        )
