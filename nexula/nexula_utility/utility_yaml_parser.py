import yaml


def load_yaml(file):
    """
    Parse yaml file into dictionary
    Parameters
    ----------
    file

    Returns
    -------

    """
    with open(file, 'r+') as file:
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    return input_data
