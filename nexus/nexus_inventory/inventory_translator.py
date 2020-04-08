NEXUS_INVENTORY_PREPROCESSER = {
    'nexus_basic_preprocesser' : ('nexus.nexus_inventory.inventory_data_preprocesser', 'NexusBasicPreprocesserData')
}

NEXUS_INVENTORY_DATA_READER = {
    'read_csv' : ('nexus.nexus_data.data_reader', 'read_csv') # TODO : change to inventory
}

NEXUS_INVENTORY_FEATURE_REPRESENTER = {
    'nexus_tf_idf_representer' : ('nexus.nexus_inventory.inventory_feature_representer',
                                  'NexusTfIdfSklearnRepresenterData')
}

NEXUS_INVENTORY_POSTPROCESSER = {

}

NEXUS_INVENTORY_MODEL = {
    'nexus_boomer_logistic_regression' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                          'BoomerLogisticRegression'),
    'nexus_boomer_linear_svc' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                          'BoomerLinearSVC'),
    'nexus_boomer_gaussian_process' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                          'BoomerGaussianProcess'),
    'nexus_boomer_random_forest' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                          'BoomerRandomForest'),
    'nexus_boomer_ada_boost' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                          'BooomerAdaBoost'),
    'nexus_boomer_multinomial_nb' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                          'BoomerMultinomialNB'),
    'nexus_boomer_quadratic_discriminant' : ('nexus.nexus_inventory.inventory_model.model_boomer',
                                                'BoomerQuadraticDiscriminant'),
}

NEXUS_INVENTORY_METRICS = {
    'acc' : ('nexus.nexus_inventory.inventory_metrics','nexus_accuracy_score'),
    'f1_macro' : ('nexus.nexus_inventory.inventory_metrics','nexus_f1_score_macro'),
    'f1_micro' : ('nexus.nexus_inventory.inventory_metrics','nexus_f1_score_micro')
}

NEXUS_CALLBACKS = {
    'model_saver_callback' : ('nexus.nexus_inventory.inventory_callback.callback_general', 'ModelSaverCallback'),
    'benchmark_reporter_callback' : ('nexus.nexus_inventory.inventory_callback.callback_general',
                                     'BenchmarkReporterCallback'),
}