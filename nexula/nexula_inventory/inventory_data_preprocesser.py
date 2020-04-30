import logging

import numpy as np
import pandas as pd

from nexula.nexula_inventory.inventory_base import NexusBaseDataInventory

logger = logging.getLogger("nexula.preprocessing")


class NexusBasicPreprocesserData(NexusBaseDataInventory):

    def __init__(self, operations=['lowercase'], **kwargs):
        super().__init__(**kwargs)
        self.operations = operations

    def nexus_lowercaser(self, x: np.array, y: np.array, *args, **kwargs):
        """
        Lowercase text
        """
        x = pd.Series(x)
        return x.str.lower().values, y

    def get_model(self):
        return self.model

    def __call__(self, x, y, fit_to_data=True, *args, **kwargs):
        """
        Lowercase the text
        Parameters
        ----------
        fit_to_data: bool
            UNUSED

        """
        for operation in self.operations:
            if operation.lower().strip() == 'lowercase':
                logger.info("Do lowercase operation")
                x, y = self.nexus_lowercaser(x,y)
        return x, y
