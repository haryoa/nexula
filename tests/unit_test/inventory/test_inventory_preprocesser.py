from nexus.nexus_inventory.inventory_data_preprocesser import NexusBasicPreprocesserData


def test_tf_idf():
    preproc = NexusBasicPreprocesserData(operations=['lowercase'])
    x, _ = preproc(['I AM A MAN'],[1])
    print(x)
    assert x == 'i am a man'
