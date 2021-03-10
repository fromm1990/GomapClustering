import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Set

from pandas import DataFrame
from shapely.geometry import Point
from shapely.strtree import STRtree

import utility
from model import Position

DEFAULT_CLASSES = {
    'b11',    'c61',    'c51',    'c55:50',
    'c55:60', 'c55:70', 'c56:60', 'c56:70',
    'c62',    'd11.3',  'd15.2',  'd15.3',
    'e23',    'e33.1',  'e55',    'e56',
    'n42.2',  'n42.3'
}
DATA_PATH = Path(__file__).parents[1].joinpath('data')
AALBORG_DATA_PATH = DATA_PATH.joinpath('aalborg')
HP_TUNING_DATA_PATH = DATA_PATH.joinpath('hp_tuning')

logger = logging.getLogger(Path(__file__).stem)

def map_aal_data(position: Position):
    time_format = '%Y-%m-%dT%H:%M:%S'
    return {
            'id': position.properties['object_no'],
            'geom': Point(position),
            'trip_id': position.properties['trip_no'],
            'classifier': position.properties['label_name'],
            'speed': position.properties['speed'],
            'heading': math.radians(position.properties['obj_heading']),
            'score': position.properties['score'],
            'distance': position.properties['distance_to_object'],
            'trip_start_time': datetime.strptime(position.properties['trip_start_time'], time_format),
            'trip_stop_time': datetime.strptime(position.properties['trip_stop_time'], time_format),
            'image_capture_time': datetime.strptime(position.properties['image_capture_time'], time_format)
        }


def load_gomap_detections(filepath: str, classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame:
    detections = utility.load_geoJson(filepath, 'epsg:4326', 'epsg:3044')
    df = DataFrame(map(map_aal_data, detections))
    logger.info('Loaded %d detections', len(df))

    # Cleansing
    df = df[df['classifier'].isin(classes)]
    logger.info('%d detections remains after applying class filter', len(df))
    logger.debug(df.groupby('classifier')['classifier'].agg('count'))

    df = df[df['score'] >= min_score]
    logger.info('%d detections remains after applying min_score >= %f filter', len(df), min_score)

    df = df[df['distance'] <= max_distance]
    logger.info('%d detections remains after applying max_distance <= %f meter filter', len(df), max_distance)
    logger.debug(df.groupby('classifier')['classifier'].agg('count'))
    return df


def load_aal_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(AALBORG_DATA_PATH.joinpath('traffic_sign_detections.geojson'), classes, min_score, max_distance)


def load_gomap_train_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(HP_TUNING_DATA_PATH.joinpath('training.geojson'), classes, min_score, max_distance)


def load_gomap_validation_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(HP_TUNING_DATA_PATH.joinpath('validation.geojson'), classes, min_score, max_distance)


def load_gomap_test_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(HP_TUNING_DATA_PATH.joinpath('testing.geojson'), classes, min_score, max_distance)


def _map_aal_truth(position: Position):
    position.properties['heading'] = math.radians(position.properties['map_heading'])
    return position


def load_gomap_truths(filepath: str, classes: Set[str] = DEFAULT_CLASSES) -> STRtree:
    def normalize_label_name(position: Position) -> Position:
        label = position.properties['label_name']
        if label in {'c55', 'c56'}:
            position.properties['label_name'] = f'{label}:{position.properties["sign_text"]}'
        return position

    truths = utility.load_geoJson(filepath, 'epsg:4326', 'epsg:3044')
    truths = list(truths)
    logger.info('Loaded %d truths', len(truths))

    truths = map(normalize_label_name, truths)
    truths = [x for x in truths if x.properties['label_name'] in classes]
    logger.info('%d truths remains after applying class filter', len(truths))

    truths = map(_map_aal_truth, truths)

    return STRtree(truths)


def load_aal_truths(classes: Set[str] = DEFAULT_CLASSES) -> STRtree:
    return load_gomap_truths(AALBORG_DATA_PATH.joinpath('traffic_sign_ground_truth.geojson'), classes)


def load_gomap_test_truths(classes: Set[str] = DEFAULT_CLASSES) -> STRtree:
    return load_gomap_truths(HP_TUNING_DATA_PATH.joinpath('testing_truth.geojson'), classes)
