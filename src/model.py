from shapely.geometry import Point
from typing import Any, Dict


class Position(Point):
    properties: Dict[str, Any]

    def __init__(self, *args):
        super().__init__(*args)
        self.visited = False

        if len(args) == 1 and isinstance(args[0], Position):
            self.properties = args[0].properties.copy()
        else:
            self.properties = {}

    def __hash__(self):
        return id(self)


class Prediction(Position):
    def __init__(self, prediction: Position, truth: Position = None):
        super().__init__(prediction)
        self.truth = truth
