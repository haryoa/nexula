import importlib


def import_class(module_name, func_name):
    """
    Import class manually and get the module. The module can be instantiated later on.
    Parameters
    ----------
    module_name
    func_name

    Returns
    -------

    """
    return getattr(importlib.import_module(module_name), func_name)
