import math

import data_loader
import numpy as np
import utility
from evaluator import GoMapEvaluator
from GoMapClustering import AngleBalancingDBSCAN


def test_debug():
    detections = data_loader.load_aal_detections()
    truths = data_loader.load_aal_truths()
    c = AngleBalancingDBSCAN()
    predictions = utility.get_predictions(detections, c, utility.compute_cluster_centroid)
    evalutator = GoMapEvaluator(predictions, truths).evaluate()
