from pathlib import Path

import data_loader
from GoMapClustering import DBSCAN
from optuna.trial import Trial

from HpOptimization.gomap_study import GoMapStudy

projec_dir = Path(__file__).parent.parent.parent
print(projec_dir)
training_data = data_loader.load_gomap_detections(
    projec_dir / 'data' / 'hp_tuning' / 'training.geojson'
)
training_truth_data = data_loader.load_gomap_truths(
    projec_dir / 'data' / 'hp_tuning' / 'training_truth.geojson'
)

validation_data = data_loader.load_gomap_detections(
    projec_dir / 'data' / 'hp_tuning' / 'validation.geojson'
)
validation_truth_data = data_loader.load_gomap_truths(
    projec_dir / 'data' / 'hp_tuning' / 'validation_truth.geojson'
)

test_data = data_loader.load_gomap_detections(
    projec_dir / 'data' / 'hp_tuning' / 'testing.geojson'
)
test_truth_data = data_loader.load_gomap_truths(
    projec_dir / 'data' / 'hp_tuning' / 'testing_truth.geojson'
)


def spawner(trial: Trial):
    max_distance = trial.suggest_float('max_distance', 0.1, 100, step=0.1)
    min_samples = trial.suggest_int('min_samples', 0, 20)

    return DBSCAN(eps=max_distance, min_samples=min_samples)


study = GoMapStudy(
    'DBSCAN',
    spawner,
    training_data,
    validation_data,
    test_data,
    training_truth_data,
    validation_truth_data,
    test_truth_data,
    early_stopping=1000
)
study.optimize()
