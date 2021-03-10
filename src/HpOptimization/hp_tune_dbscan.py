from pathlib import Path

import data_loader
from GoMapClustering import DBSCAN
from optuna.trial import Trial

from HpOptimization.gomap_study import GoMapStudy

data_dir = Path.cwd() / 'data' / 'hp_tuning'

# Training
training_data = data_loader.load_gomap_detections(
    data_dir / 'training.geojson'
)
training_truth_data = data_loader.load_gomap_truths(
    data_dir / 'training_truth.geojson'
)

# Validation
validation_data = data_loader.load_gomap_detections(
    data_dir / 'validation.geojson'
)
validation_truth_data = data_loader.load_gomap_truths(
    data_dir / 'validation_truth.geojson'
)

# Testing
test_data = data_loader.load_gomap_detections(
    data_dir / 'testing.geojson'
)
test_truth_data = data_loader.load_gomap_truths(
    data_dir / 'testing_truth.geojson'
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
