import logging
import math
from logging import FileHandler, Logger, StreamHandler
from typing import Callable

import utility
import numpy as np
import optuna
import plotly.express as px
import wandb
from evaluator import GoMapEvaluator
from GoMapClustering.base import GoMapClusterMixin
from optuna import Study
from optuna.trial import FrozenTrial, Trial
from optuna.visualization import (plot_parallel_coordinate,
                                  plot_param_importances)
from pandas.core.frame import DataFrame
from shapely.strtree import STRtree


class GoMapStudy(Study):
    
    def __init__(
        self,
        study_name: str,
        spawner: Callable[[Trial], GoMapClusterMixin],
        training: DataFrame,
        validation: DataFrame,
        testing: DataFrame,
        training_truth: STRtree,
        validation_truth: STRtree,
        testing_truth: STRtree,
        logfile: str = None,
        early_stopping=0
    ):
        wandb.init(project="gomap-clustering", name=study_name)
        study = optuna.create_study(study_name=study_name)
        super().__init__(
            study.study_name,
            study._storage,
            study.sampler,
            study.pruner
        )

        self.spawner = spawner

        # Training
        self.training = training
        self.training_truth = training_truth

        # Validation
        self.validation = validation
        self.validation_truth = validation_truth

        # Testing
        self.testing = testing
        self.testing_truth = testing_truth

        # Losses
        self.training_loss = []
        self.validation_loss = []
        
        self.early_stopping = early_stopping
        self.logger = self.__get_logger(logfile)


    @property
    def best_training_trial(self) -> FrozenTrial:
        min_training_loss_idx = self.training_loss.index(min(self.training_loss))
        return self.get_trials(False)[min_training_loss_idx]


    @property
    def best_validation_trial(self) -> FrozenTrial:
        min_validation_loss_idx = self.validation_loss.index(min(self.validation_loss))
        return self.get_trials(False)[min_validation_loss_idx]


    def optimize(self):
        callbacks = [self.__best_result_logger, self.__loss_logger]

        if self.early_stopping > 0:
            callbacks.append(self.__early_stopping)

        super().optimize(self.__objective, callbacks=callbacks)


    def stop(self) -> None:
        val_eval = self.__get_evaluation(self.validation, self.validation_truth, self.best_validation_trial)
        train_eval = self.__get_evaluation(self.training, self.training_truth, self.best_training_trial)
        test_eval = self.__get_evaluation(self.testing, self.testing_truth, self.best_validation_trial)

        wandb.config.update({
            'params': self.best_validation_trial.params
        })

        wandb.summary.update({
            'f1': test_eval.f1,
            'recall': test_eval.recall,
            'precision': test_eval.precision,
            'rmse_angle': test_eval.rmse_direction_degrees,
            'rmse_location': test_eval.rmse_location,
            'mae_angle': test_eval.mae_direction_degrees,
            'mae_location': test_eval.mae_location
        })

        wandb.log({
            'parallel_coordinate': plot_parallel_coordinate(self),
            'param_importances': plot_param_importances(self),
            'angle_error_boxplot': self.__plot_angle_error_box(val_eval, train_eval, test_eval),
            'location_error_boxplot': self.__plot_location_error_box(val_eval, train_eval, test_eval)
        })

        return super().stop()


    @staticmethod
    def __add_stream_handler(logger: Logger) -> None:
        # Check if such handler is already present
        for handler in logger.handlers:
            if isinstance(handler, StreamHandler):
                return
        return logger.addHandler(StreamHandler())


    @staticmethod
    def __add_file_handler(logger: Logger, filepath: str) -> None:
        # Check if such handler is already present
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                return
        return logger.addHandler(FileHandler(filepath))


    @staticmethod
    def __plot_angle_error_box(
        val_eval: GoMapEvaluator.Evaluation,
        train_eval: GoMapEvaluator.Evaluation,
        test_eval: GoMapEvaluator.Evaluation
    ):
        # TODO: Check if this is a possible way to solve
        val_errors = DataFrame(val_eval.errors['direction_error'], columns=['angle_error'])
        val_errors['type'] = 'validation'

        train_errors = DataFrame(train_eval.errors['direction_error'], columns=['angle_error'])
        train_errors['type'] = 'training'

        test_errors = DataFrame(test_eval.errors['direction_error'], columns=['angle_error'])
        test_errors['type'] = 'testing'

        
        df = val_errors.append([train_errors, test_errors])
        return px.box(df, x="type", y="angle_error")

        # values = np.concatenate((
        #     np.degrees(val_eval.errors['direction_error']),
        #     np.degrees(train_eval.errors['direction_error']),
        #     np.degrees(test_eval.errors['direction_error'])
        # ))
        # df = DataFrame(values, columns=['angle_error'])
        # df['type'] = np.concatenate((
        #     np.full(len(val_eval.errors['direction_error']), 'validation'),
        #     np.full(len(train_eval.errors['direction_error']), 'training'),
        #     np.full(len(test_eval.errors['direction_error']), 'testing')
        # ))
        
        # return px.box(df, x="type", y="angle_error")


    @staticmethod
    def __plot_location_error_box(
        val_eval: GoMapEvaluator.Evaluation,
        train_eval: GoMapEvaluator.Evaluation,
        test_eval: GoMapEvaluator.Evaluation
    ):
        val_errors = DataFrame(val_eval.errors['location_error'], columns=['location_error'])
        val_errors['type'] = 'validation'

        train_errors = DataFrame(train_eval.errors['location_error'], columns=['location_error'])
        train_errors['type'] = 'training'

        test_errors = DataFrame(test_eval.errors['location_error'], columns=['location_error'])
        test_errors['type'] = 'testing'

        df = val_errors.append([train_errors, test_errors])
        return px.box(df, x="type", y="location_error")

        # values = np.concatenate((
        #     val_eval.location_errors,
        #     train_eval.location_errors,
        #     test_eval.location_errors
        # ))
        # df = DataFrame(values, columns=['location_error'])
        # df['type'] = np.concatenate((
        #     np.full(len(val_eval.location_errors), 'validation'),
        #     np.full(len(train_eval.location_errors), 'training'),
        #     np.full(len(test_eval.location_errors), 'testing')
        # ))

        # return px.box(df, x="type", y="location_error")


    def __get_evaluation(self, data, truths, trial) -> GoMapEvaluator.Evaluation:
        predictions = utility.get_predictions(
            data,
            self.spawner(trial),
            utility.compute_cluster_centroid
        )
        return GoMapEvaluator(predictions, truths).evaluate()


    def __get_logger(self, logfile: str = None) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if logfile is not None:
            self.__add_file_handler(logger, logfile)
        self.__add_stream_handler(logger)

        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        return logger
    

    @staticmethod
    def __get_loss(evaluator: GoMapEvaluator.Evaluation) -> float:
        return 1 - evaluator.f1


    def __validation_objective(self, trial: FrozenTrial) -> None:
        evaluation = self.__get_evaluation(self.validation, self.validation_truth, trial)

        wandb.log({
            'validation': {
                'mae_angle': evaluation.mse_direction_degrees,
                'mae_location': evaluation.mae_location,
                'rmse_angle': evaluation.rmse_direction_degrees,
                'rmse_location': evaluation.rmse_location,
                'recall': evaluation.recall,
                'precision': evaluation.precision,
                'f1': evaluation.f1
            }
        })
        self.validation_loss.append(self.__get_loss(evaluation))

    
    def __training_objective(self, trial: FrozenTrial) -> None:
        evaluation = self.__get_evaluation(self.training, self.training_truth, trial)

        wandb.log({
            'training': {
                'mae_angle': evaluation.mse_direction_degrees,
                'mae_location': evaluation.mae_location,
                'rmse_angle': evaluation.rmse_direction_degrees,
                'rmse_location': evaluation.rmse_location,
                'recall': evaluation.recall,
                'precision': evaluation.precision,
                'f1': evaluation.f1
            }
        })
        self.training_loss.append(self.__get_loss(evaluation))


    def __objective(self, trial: Trial) -> float:
        self.__validation_objective(trial)
        self.__training_objective(trial)

        return self.training_loss[-1]


    def __early_stopping(self, study: Study, trial: FrozenTrial):
        if trial.number - self.best_validation_trial.number > self.early_stopping:
            study.stop()


    def __best_result_logger(self, study: Study, trial: FrozenTrial):
        if study.best_trial == trial:
            self.logger.info(f'Trial: {trial.number}, Parameters: {trial.params}, loss: {trial.value}')


    def __loss_logger(self, study: Study, trial: FrozenTrial):
        wandb.log({
            'training_loss': self.training_loss[-1],
            'validation_loss': self.validation_loss[-1]
        })
