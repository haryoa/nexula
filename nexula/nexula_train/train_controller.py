import logging
from typing import List

from nexula.nexula_data.data_loader import BaseNexusDataLoader
from nexula.nexula_inventory.inventory_base import NexusBaseCallbackInventory
from nexula.nexula_inventory.inventory_base import NexusBaseModelInventory

logger = logging.getLogger("nexula.trainer")


class TrainerController:
    """
    Controller which handles training
    """

    def __init__(self, data_choice_type, models: List[NexusBaseModelInventory], models_args, data_choice_args={},
                 callbacks: List[NexusBaseCallbackInventory] = []):

        # TODO Refactor to be similar to data controller
        self.data_choice_type = data_choice_type
        if data_choice_type == 'manual_split':
            self.run = self.run_default
            self._init_data_choice_default(**data_choice_args)
        self.models = models
        self.model_args = models_args
        self.callbacks = callbacks
        self.models_obj = []

    def _init_data_choice_default(self, train_dataloader: BaseNexusDataLoader, validation_dataloader: BaseNexusDataLoader,
                                    test_dataloader: BaseNexusDataLoader):
        """
        Construct the variable if the `data_choice_type` is 'manual_split"

        """
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def run_default(self, *args, **kwargs):
        """
        Run training on 'manual_split' choice. Also run the call back on several steps.

        """
        # TODO REFACTOR
        # STEP before_loop_step
        self.run_callback('before_loop_step')
        self.preprocess_model_args()  # Preprocess for millenial
        for i in range(len(self.models)):
            model, model_args = self.models[i], self.model_args[i]
            # model = import_class(*nexus_inventory_model[model])
            model_obj: NexusBaseModelInventory = model(model_args)
            self.models_obj.append(model_obj)
            # STEP in_loop_before_fit_step
            logger.info("Run {} Model".format(model_obj.__class__.__name__))
            self.run_callback('in_loop_before_fit_step')
            model_obj.fit_to_dataloader(self.train_dataloader,
                                        val_dataloader=self.validation_dataloader,
                                        test_dataloader=self.test_dataloader)
            self.run_callback('in_loop_after_fit_step')
            # STEP in_loop_after_fit_step
        self.run_callback('after_loop_step')
        # STEP after_loop_step

    def preprocess_model_args(self):
        """ Add info about pretrained model and num_label in millenial"""
        for i, model in enumerate(self.models):
            if model.type == 'millenial':
                self.model_args[i].setdefault('additional_info', dict())
                feature_obj = self.train_dataloader.nexus_feature_representer.list_of_cls[-1]
                text_field = feature_obj.text_field
                self.model_args[i]['additional_info']['text_field'] = text_field
                self.model_args[i]['additional_info']['label_dist'] = self.train_dataloader.label_dist

    def run_callback(self, step):
        for callback in self.callbacks:
            if step == 'before_loop_step':
                callback.before_loop_step(self)
            elif step == 'in_loop_before_fit_step':
                callback.in_loop_before_fit_step(self)
            elif step == 'in_loop_after_fit_step':
                callback.in_loop_after_fit_step(self)
            elif step == 'after_loop_step':
                callback.after_loop_step(self)
