import pytest
import types


def test_import_var():
    """
    Check import function
    """
    from nexus.nexus_utility.utility_extract_func import import_class
    imported_var = import_class('nexus.nexus_utility.utility_extract_func', 'import_class')
    assert type(imported_var) == types.FunctionType
