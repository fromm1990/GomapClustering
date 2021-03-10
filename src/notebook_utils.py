from typing import Callable, List

from pandas import DataFrame, Series

from GoMapClustering.base import GoMapClusterMixin
from GoMapClustering.utility import TWO_PI, AngleIndexer
from model import Position


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
