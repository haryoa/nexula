import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger('nexula.data_reader')


def read_csv(file: str, x_column='text', y_column='label', *args, **kwargs) -> Tuple[np.array, np.array]:
    """
    Read csv file
    Parameters
    ----------
    file: str
        File name
    x_column: str
        Column which act as text
    y_column: str
        Column which act as label

    Returns
    -------
    np.array, np.array
        text and label as `np.array` respectively
    """
    df = pd.read_csv(file)
    logger.debug("Read {} csv file".format(file))
    x_data = df[x_column].values
    y_data = df[y_column].values
    return x_data, y_data
