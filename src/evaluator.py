import math
from typing import Callable, List, Tuple, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from shapely.geometry import Point
from shapely.strtree import STRtree

import utility
from model import Position


class GoMapEvaluator:
    __angle_errors: ndarray = None
    __location_errors: ndarray = None
    __norm_angle_errors: ndarray = None
    __norm_location_errors: ndarray = None
    __precision: float = None
    __recall: float = None
    __f1: float = None

    def __init__(self, predictions: DataFrame, truths: STRtree):
        self.truths = truths
        self.predictions = predictions.assign(truth=self.__get_truths(predictions))


    @property
    def angle_errors(self) -> ndarray:
        if self.__angle_errors is not None:
            return self.__angle_errors

        def get_truth_heading(truth: Point):
            if truth is None:
                return None
            return truth.properties['heading']

        get_angle_errors: Callable[[ndarray, ndarray], ndarray] = np.vectorize(self.__get_angle_error, otypes=[float])
        self.__angle_errors = get_angle_errors(
            self.predictions['heading'],
            self.predictions['truth'].apply(get_truth_heading)
        )

        return self.__angle_errors


    @property
    def norm_angle_errors(self) -> ndarray:
        if self.__norm_angle_errors is not None:
            return self.__norm_angle_errors

        if(self.angle_errors.size == 0):
            self.__norm_angle_errors = np.empty(0)
        else:
            self.__norm_angle_errors = self.angle_errors / math.radians(180)

        return self.__norm_angle_errors


    @property
    def mse_angle(self) -> float:
        return np.nanmean(self.angle_errors**2)


    @property
    def rmse_angle(self) -> float:
        return math.sqrt(self.mse_angle)


    @property
    def mae_angle(self) -> float:
        return np.nanmean(self.angle_errors)


    @property
    def norm_mse_angle(self) -> float:
        return np.nanmean(self.norm_angle_errors**2)


    @property
    def location_errors(self) -> ndarray:
        if self.__location_errors is not None:
            return self.__location_errors
        
        get_location_errors: Callable[[ndarray, ndarray], ndarray] = np.vectorize(self.__get_location_error, otypes=[float])
        self.__location_errors = get_location_errors(
            self.predictions['geom'],
            self.predictions['truth']
        )
        return self.__location_errors


    @property
    def norm_location_errors(self) -> ndarray:
        if self.__norm_location_errors is not None:
            return self.__norm_location_errors
        
        if self.location_errors.size == 0:
            self.__norm_location_errors = np.empty(0)
        else:
            self.__norm_location_errors = self.__location_errors / 10

        return self.__norm_location_errors


    @property
    def mse_location(self) -> float:
        return np.nanmean(self.location_errors**2)


    @property
    def rmse_location(self) -> float:
        return math.sqrt(self.mse_location)

    
    @property
    def mae_location(self) -> float:
        return np.nanmean(self.location_errors)


    @property
    def norm_mse_location(self):
        return np.nanmean(self.norm_location_errors**2)


    @property
    def precision(self):
        if self.__precision is not None:
            return self.__precision

        true_positives = len(self.predictions[self.predictions['truth'].notnull()])
        false_positives = len(self.predictions) - true_positives

        denominator = true_positives + false_positives
        self.__precision = math.nan if denominator == 0 else true_positives / denominator
        return self.__precision


    @property
    def recall(self):
        if self.__recall is not None:
            return self.__recall

        true_positives = len(self.predictions[self.predictions['truth'].notnull()])
        false_negatives = len(self.truths._geoms) - true_positives

        denominator = true_positives + false_negatives
        self.__recall = math.nan if denominator == 0 else true_positives / denominator
        return self.__recall
    

    @property
    def f1(self):
        if self.__f1 is not None:
            return self.__f1

        denominator = self.precision + self.recall
        if math.isnan(denominator) or denominator == 0:
            return math.nan

        self.__f1 = 2 * (self.precision * self.recall / denominator)
        return self.__f1


    @staticmethod
    def __get_angle_error(prediction: float, truth: float) -> Union[float, None]:
        if prediction is None or truth is None:
            return None

        return abs(utility.signed_angle_diff(prediction, truth))


    @staticmethod
    def __get_location_error(prediction: Point, truth: Point) -> Union[float, None]:
        if prediction is None or truth is None:
            return None

        return prediction.distance(truth)


    def __get_candidate_errors(self, cluster_center: Point, heading: float, candidates: List[Position]) -> Tuple[float, Position]:
        norm_location_errors = (self.__get_location_error(cluster_center, x) / 20 for x in candidates)
        norm_angle_errors = (self.__get_angle_error(heading, x.properties['heading']) / 180 for x in candidates)
        norm_errors = (x + y for x, y in zip(norm_location_errors, norm_angle_errors))
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


    def __get_truths(self, predictions: DataFrame) -> ndarray:
        data = zip(
            predictions.index.get_level_values('classifier'),
            predictions['geom'],
            predictions['heading']
        )

        visited = set()
        def get_truth(classifier: str, cluster_center: Point, heading: float) -> Union[Point, None]:
            for candidate in self.__get_truth_candidates(classifier, cluster_center, heading):
                if candidate not in visited:
                    visited.add(candidate)
                    return candidate
            return None

        return np.array([
            get_truth(classifier, cluster_center, heading) 
            for classifier, cluster_center, heading 
            in data
        ], dtype=np.object)
