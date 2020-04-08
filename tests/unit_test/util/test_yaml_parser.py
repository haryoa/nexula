import pytest


def test_yaml_parser():
    from nexus.nexus_utility.utility_yaml_parser import load_yaml
    dictionary = load_yaml('tests/dummy_data/test.yaml')
    assert 'name' in dictionary and 'job' in dictionary
    assert dictionary['name'] == 'haryo'
    assert len(dictionary['job']) == 2