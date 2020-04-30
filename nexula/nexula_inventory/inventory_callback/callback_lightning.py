from nexula.nexula_inventory.inventory_base import NexusBaseCallbackInventory
from nexula.nexula_inventory.inventory_model.model_millenial import MillenialBaseClassification
from nexula.nexula_train.train_controller import TrainerController
from nexula.nexula_utility.utility_import_var import import_class


class LightningCallback(NexusBaseCallbackInventory):
    def __init__(self, list_callback=[], *args, **kwargs):
        self.list_callback = list_callback
        self.obj_callback = []
        for dict_callback in list_callback:
            self.obj_callback.append(self._translate_callbacks(dict_callback))

    def _translate_callbacks(self, dict_callback):
        callback = dict_callback['callback']
        param_callback = dict_callback['params']
        callback_class = import_class('pytorch_lightning.callbacks', callback)
        return callback_class(**param_callback)

    def before_loop_step(self, trainer_object):
        pass

    def after_loop_step(self, trainer_object):
        pass

    def in_loop_before_fit_step(self, trainer_object):
        current_model: MillenialBaseClassification = trainer_object.models_obj[-1]
        current_model.extend_callbacks(self.obj_callback)

    def in_loop_after_fit_step(self, trainer_object: TrainerController):
        pass
