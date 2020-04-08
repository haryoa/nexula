import pandas as pd
from typing import Tuple
import numpy as np


def read_csv(file: str, x_column='text', y_column='label', *args, **kwargs) -> Tuple[np.array, np.array]:
    """
    Read csv
    Parameters
    ----------
    file
    x_column
    y_column
    args
    kwargs

    Returns
    -------

    """
    df = pd.read_csv(file)
    x_data = df[x_column].values
    y_data = df[y_column].values
    return x_data, y_data
