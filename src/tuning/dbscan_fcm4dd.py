from pathlib import Path

import data_loader
from GoMapClustering import DBSCANFCM4DD
from optuna.trial import Trial

from tuning.gomap_study import GoMapStudy

data_dir = Path() / 'data' / 'hp_tuning'

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
    c = trial.suggest_int('c', 2, 10)
    m = trial.suggest_discrete_uniform('m', 2, 10, 0.1)
    max_iteratians = trial.suggest_int('max_iteratians', 10, 200)
    min_improvement = trial.suggest_uniform('min_improvement', 1e-9, 1)
    max_distance = trial.suggest_discrete_uniform('max_distance', 0.1, 100, 0.1)
    min_samples = trial.suggest_int('min_samples', 0, 20)

    return DBSCANFCM4DD(
        c=c,
        m=m,
        max_iterations=max_iteratians,
        min_improvement=min_improvement,
        max_spatial_distance=max_distance,
        min_samples=min_samples,
        seed=1337
    )

study = GoMapStudy(
    'DBSCAN + FCM4DD',
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
