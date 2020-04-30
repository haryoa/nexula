from nexula.nexula_inventory.inventory_feature_representer import NexusTfIdfSklearnRepresenterData


def test_tf_idf():
    preproc = NexusTfIdfSklearnRepresenterData()
    x, _ = preproc(['TEST','TESTIS'],[1,1])
    dim_0 = x[0,0]
    dim_1 = x[1,1]
    assert dim_0 == 1
    assert dim_1 == 1
