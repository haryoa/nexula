from nexus.nexus_data.data_loader import BaseNexusDataLoader
from nexus.nexus_inventory.inventory_base import NexusBaseModelInventory
from typing import List
from nexus.nexus_inventory.inventory_base import NexusBaseCallbackInventory


class TrainerController:

    def __init__(self, data_choice_type, models, models_args, data_choice_args = {},
                 callbacks:List[NexusBaseCallbackInventory]=[]):

        # TODO Refactor to be similar to data controller
        self.data_choice_type = data_choice_type
        if data_choice_type == 'manual_split':
            self.run = self.run_default
            self._init_data_choice_default(**data_choice_args)
        self.models = models
        self.models_args = models_args
        self.callbacks = callbacks
        self.models_obj = []

    def _init_data_choice_default(self, train_dataloader: BaseNexusDataLoader, validation_dataloader: BaseNexusDataLoader,
                                    test_dataloader: BaseNexusDataLoader):
        """
        Construct the variable if the `data_choice_type` is 'manual_split"
        Parameters
        ----------
        train_dataloader
        validation_dataloader
        test_dataloader

        Returns
        -------

        """
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def run_default(self, *args, **kwargs):
        """
        Run training on 'manual_split' choice. Also run the call back on several steps.

        Parameters
        ----------
        train_dataloader
        validation_dataloader
        test_dataloader

        Returns
        -------

        """
        # TODO REFACTOR
        # STEP before_loop_step
        self.run_callback('before_loop_step')
        for i in range(len(self.models)):
            model, model_args = self.models[i], self.models_args[i]
            # model = import_class(*nexus_inventory_model[model])
            model_obj:NexusBaseModelInventory = model(model_args)
            self.models_obj.append(model_obj)
            # STEP in_loop_before_fit_step
            self.run_callback('in_loop_before_fit_step')
            model_obj.fit_to_dataloader(self.train_dataloader)
            self.run_callback('in_loop_after_fit_step')
            # STEP in_loop_after_fit_step
        self.run_callback('after_loop_step')
        # STEP after_loop_step

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
