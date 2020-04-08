from nexus.nexus_controller.nexus_core import NexusController
import os
import json


def test_run_sample():
    args = ['-r','tests/dummy_data/sample_run.yaml']
    controller = NexusController()
    controller.test_main(args)
    listdir = os.listdir('output/integration_test/')
    assert 'BoomerLogisticRegression' in listdir

    with open('output/integration_test/report.json', 'r+') as file:
        json_read = json.load(file)

    assert 'BoomerLogisticRegression' in json_read
    listdir = os.listdir('output/integration_test/BoomerLogisticRegression')
    assert 'finalized_model.sav' in listdir
