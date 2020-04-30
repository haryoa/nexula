NEXUS_INVENTORY_PREPROCESSER = {
    'nexus_basic_preprocesser' : ('nexula.nexula_inventory.inventory_data_preprocesser', 'NexusBasicPreprocesserData')
}

NEXUS_INVENTORY_DATA_READER = {
    'read_csv' : ('nexula.nexula_data.data_reader', 'read_csv') # TODO : change to inventory
}

NEXUS_INVENTORY_FEATURE_REPRESENTER = {
    'nexus_tf_idf_representer': ('nexula.nexula_inventory.inventory_feature_representer',
                                 'NexusTfIdfSklearnRepresenterData'),
    'nexus_millenial_representer': ('nexula.nexula_inventory.inventory_feature_representer',
                                    'NexusMillenialTorchTextRepresenter')
}

NEXUS_INVENTORY_POSTPROCESSER = {

}

NEXUS_INVENTORY_MODEL = {
    'nexus_boomer_logistic_regression' : ('nexula.nexula_inventory.inventory_model.model_boomer',
                                          'BoomerLogisticRegression'),
    'nexus_boomer_linear_svc' : ('nexula.nexula_inventory.inventory_model.model_boomer',
                                          'BoomerLinearSVC'),
    'nexus_boomer_gaussian_process': ('nexula.nexula_inventory.inventory_model.model_boomer',
                                      'BoomerGaussianProcess'),
    'nexus_boomer_random_forest': ('nexula.nexula_inventory.inventory_model.model_boomer',
                                   'BoomerRandomForest'),
    'nexus_boomer_ada_boost': ('nexula.nexula_inventory.inventory_model.model_boomer',
                               'BooomerAdaBoost'),
    'nexus_boomer_multinomial_nb': ('nexula.nexula_inventory.inventory_model.model_boomer',
                                    'BoomerMultinomialNB'),
    'nexus_boomer_quadratic_discriminant': ('nexula.nexula_inventory.inventory_model.model_boomer',
                                            'BoomerQuadraticDiscriminant'),
    'nexus_millenial_ccn1d_classification': ('nexula.nexula_inventory.inventory_model.model_millenial',
                                             'MillenialCNN1DClassification'),
    'nexus_millenial_lstm_classification': ('nexula.nexula_inventory.inventory_model.model_millenial',
                                            'MillenialLSTMClassification'),
}

NEXUS_INVENTORY_METRICS = {
    'acc' : ('nexula.nexula_inventory.inventory_metrics','nexus_accuracy_score'),
    'f1_macro': ('nexula.nexula_inventory.inventory_metrics', 'nexus_f1_score_macro'),
    'f1_micro': ('nexula.nexula_inventory.inventory_metrics', 'nexus_f1_score_micro')
}

NEXUS_CALLBACKS = {
    'model_saver_callback': ('nexula.nexula_inventory.inventory_callback.callback_general', 'ModelSaverCallback'),
    'benchmark_reporter_callback': ('nexula.nexula_inventory.inventory_callback.callback_general',
                                    'BenchmarkReporterCallback'),
    'lightning_callback': ('nexula.nexula_inventory.inventory_callback.callback_lightning', 'LightningCallback')
}


def collect_dict():
    new_dict = {}
    # can be refactored
    new_dict.update(NEXUS_INVENTORY_PREPROCESSER)
    new_dict.update(NEXUS_INVENTORY_DATA_READER)
    new_dict.update(NEXUS_INVENTORY_FEATURE_REPRESENTER)
    new_dict.update(NEXUS_INVENTORY_POSTPROCESSER)
    new_dict.update(NEXUS_INVENTORY_MODEL)
    new_dict.update(NEXUS_INVENTORY_METRICS)
    new_dict.update(NEXUS_CALLBACKS)
    return new_dict


NEXUS_INVENTORIES = collect_dict()


def update_dict_with_custom_module(new_dict):
    NEXUS_INVENTORIES.update(new_dict)
