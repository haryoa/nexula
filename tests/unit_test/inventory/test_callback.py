import json

from nexus.nexus_data.data_controller import DataController
from nexus.nexus_inventory.inventory_callback.callback_general import ModelSaverCallback, BenchmarkReporterCallback
from nexus.nexus_inventory.inventory_data_preprocesser import NexusBasicPreprocesserData
from nexus.nexus_inventory.inventory_feature_representer import NexusTfIdfSklearnRepresenterData
from nexus.nexus_inventory.inventory_model.model_boomer import BoomerRandomForest
import os
import nexus
from nexus.nexus_train.train_controller import TrainerController


def test_model_saver_callback():
    """
    Test callback
    """
    dir_path = 'output'
    dummy_x = [[0],[0],[1],[1]]
    dummy_y = [0,0,1,1]
    cb = ModelSaverCallback(dir_path)
    model = BoomerRandomForest({'call' : {}, 'init': {}})
    model.fit(dummy_x, dummy_y)
    cb.dump_model(model)
    list_dir = os.listdir('output')
    assert 'BoomerRandomForest' in list_dir
    list_dir = os.listdir('output/BoomerRandomForest')
    assert 'finalized_model.sav' in list_dir


def test_benchmark_reporter_callback():
    """
    Benchmark Reporter
    """
    data_loader_input_dict = {'data_choice_type': 'manual_split',
                              'data_reader_type': 'read_csv',
                              'data_reader_args': {'train': {'file': 'tests/dummy_data/train.csv'},
                                                   'dev': {'file': 'tests/dummy_data/dev.csv'},
                                                   'test': {'file': 'tests/dummy_data/test.csv'}},
                              'data_preprocesser_func_list_and_args': [{'process': NexusBasicPreprocesserData,
                                                                        'params': {'init': {}, 'call': {}}}],
                              'data_representer_func_list_and_args': [{'process': NexusTfIdfSklearnRepresenterData,
                                                                       'params': {'init': {}, 'call': {}}}],
                              'data_postprocesser_proc': []}

    dc = DataController(data_loader_input_dict['data_choice_type'])
    train_dataloader, validation_dataloader, test_dataloader = dc.construct_data_loader(**data_loader_input_dict)
    models = [BoomerRandomForest]
    args = [{
        'init':{},
        'call':{}
    }]
    tc = TrainerController('manual_split', models, args, {
        'train_dataloader' : train_dataloader,
        'validation_dataloader' : validation_dataloader,
        'test_dataloader' : test_dataloader
    })
    current_model = models[0](args[0])
    current_model.fit_to_dataloader(validation_dataloader)
    tc.models_obj.append(current_model)
    cb = BenchmarkReporterCallback('output')
    cb.in_loop_before_fit_step(tc)
    cb.in_loop_after_fit_step(tc)
    list_dir = os.listdir('output')
    assert 'report.json' in list_dir
    with open('output/report.json','r+') as file:
        report = json.load(file)
    assert 'BoomerRandomForest' in report