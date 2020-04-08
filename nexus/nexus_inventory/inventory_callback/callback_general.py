from nexus.nexus_inventory.inventory_base import NexusBaseCallbackInventory, NexusBaseModelInventory
from nexus.nexus_data.data_loader import BaseNexusDataLoader
from pathlib import Path
import os
from nexus.nexus_train.train_controller import TrainerController
import joblib
import json
from nexus.nexus_utility.utility_import_var import import_class
from nexus.nexus_inventory.inventory_translator import NEXUS_INVENTORY_METRICS as nexus_metrics
import timeit


class ModelSaverCallback(NexusBaseCallbackInventory):
    """
    Callback on saving the MODEL into the output directory.
    """
    def __init__(self, output_dir, *args, **kwargs):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            os.mkdir(output_dir)

    def before_loop_step(self, trainer_object):
        pass

    def after_loop_step(self, trainer_object):
        pass

    def in_loop_before_fit_step(self, trainer_object):
        pass

    def in_loop_after_fit_step(self, trainer_object: TrainerController):
        """
        Save using joblib and place it into 'output_dir/model_name/finalized_model.sav'
        Parameters
        ----------
        trainer_object

        Returns
        -------

        """
        current_model = trainer_object.models_obj[-1]
        self.dump_model(current_model)

    def dump_model(self, current_model: NexusBaseModelInventory):
        """
        Create pickle object for the trained model
        Parameters
        ----------
        current_model: NexusBaseModelInventory

        """
        if current_model.type == 'boomer':
            model_name = current_model.__class__.__name__
            output_dir = self.output_dir / model_name
            if not output_dir.is_dir():
                os.mkdir(output_dir)
            filename = 'finalized_model.sav'
            joblib.dump(current_model, output_dir / filename)


class BenchmarkReporterCallback(NexusBaseCallbackInventory):
    """
    Report the RESULT of the model in JSON
    """

    def __init__(self, output_dir, metrics = ['acc', 'f1_macro'],
                        report_dataset=['val','test'], *args, **kwargs):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            os.mkdir(output_dir)
        self.metrics = metrics
        self.report_dataset = report_dataset

    def before_loop_step(self, trainer_object):
        pass

    def after_loop_step(self, trainer_object):
        pass

    def in_loop_before_fit_step(self, trainer_object):
        self._time_start = timeit.default_timer()

    def in_loop_after_fit_step(self, trainer_object: TrainerController):

        # TODO add hyperparam into dictionary
        if trainer_object.data_choice_type == 'manual_split':
            self._manual_split_reporter(trainer_object)

    def _manual_split_reporter(self, trainer_object):
        """
        USED if the data_choice_type is manual_split
        """
        train_runtime = timeit.default_timer() - self._time_start
        report = {}
        model = trainer_object.models_obj[-1]
        model_name = model.__class__.__name__
        report[model_name] = {}
        for dataset in self.report_dataset:
            if dataset == 'val':
                dataloader = trainer_object.validation_dataloader
            elif dataset == 'test':
                dataloader = trainer_object.test_dataloader
            else:
                dataloader = trainer_object.train_dataloader
            report[model_name][dataset] = (self._report_score(dataloader, trainer_object))
        report[model_name] = self._tidy_dict(report[model_name])
        report[model_name]['model_type'] = model.type
        report[model_name]['train_runtime'] = '{} s'.format(train_runtime)
        self._write_to_json(report)

    def _tidy_dict(self, dict_report):
        """
        Fix the dictionary to be easier to see
        Parameters
        ----------
        dict_report

        Returns
        -------

        """
        returned_dict = {}
        for metric in self.metrics + ['inference_runtime']:
            returned_dict[metric] = {}

        datasets = ['val','test']

        for dt_set in datasets:
            if dt_set in dict_report:
                for metric in self.metrics + ['inference_runtime']:
                    returned_dict[metric][dt_set] = dict_report[dt_set][metric]
        return returned_dict

    def _report_score(self, dataloader:BaseNexusDataLoader, trainer_object: TrainerController):
        """
        Calculate score by using defined metrics.

        Parameters
        ----------
        dataloader
        trainer_object

        Returns
        -------

        """
        model = trainer_object.models_obj[-1]
        model_type = model.type

        dict_model = dict()

        if model_type == 'boomer':
            x_true, y_true = dataloader.get_all_processed_data(fit_to_data=False)
            dict_update = self._report_score_helper(model, x_true, y_true)
            dict_model.update(dict_update)
        return dict_model

    def _report_score_helper(self, model, x_true, y_true):
        time_start = timeit.default_timer()
        y_pred = model.predict(x_true)
        duration = timeit.default_timer() - time_start
        full_score = self._calculate_score(y_true, y_pred)
        full_score['inference_runtime'] = duration
        return full_score

    def _calculate_score(self, y_true, y_pred):
        dict_score = {}
        for metric in self.metrics:
            scorer_function = import_class(*nexus_metrics[metric])
            score = scorer_function(y_true, y_pred)
            dict_score[metric] = score
        return dict_score

    def _write_to_json(self, new_report):
        """
        Write the training report into json
        Parameters
        ----------
        new_report

        Returns
        -------

        """
        report = new_report
        file_json = Path(self.output_dir / 'report.json')
        if file_json.is_file():
            with open(file_json,'r+') as file:
                report = json.load(file)
            report.update(new_report)

        with open(file_json,'w+') as file:
            json.dump(report, file)

