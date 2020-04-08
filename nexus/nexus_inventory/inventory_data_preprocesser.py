import pandas as pd
from nexus.nexus_inventory.inventory_base import NexusBaseDataInventory
import numpy as np


class NexusBasicPreprocesserData(NexusBaseDataInventory):

    def __init__(self, operations=['lowercase'], **kwargs):
        super().__init__(**kwargs)
        self.operations = operations

    def nexus_lowercaser(self, x: np.array, y: np.array, *args, **kwargs):
        """
        Lowercase text (x)
        Parameters
        ----------
        x
        y
        args
        kwargs

        Returns
        -------

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
        x
        y
        fit_to_data
        args
        kwargs

        Returns
        -------

        """
        for operation in self.operations:
            if operation.lower().strip() == 'lowercase':
                x, y = self.nexus_lowercaser(x,y)
        return x, y
