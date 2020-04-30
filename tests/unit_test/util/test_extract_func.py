from nexula.nexula_utility.utility_extract_func import NexusFunctionModuleExtractor
from nexula.nexula_inventory.inventory_feature_representer import NexusTfIdfSklearnRepresenterData
import scipy


class TestClass():

    def __call__(self, x, y , *args, **kwargs):
        return x,y


def test_class_extractor():
    """
    Check Extractor class
    """
    class_test = [TestClass, NexusTfIdfSklearnRepresenterData]
    args = [
        {
            'init' : {},
            'call' : {}
        },
        {
            'init' : {},
            'call' : {'args_TfidfVectorizer' : {
                'binary' : True
            }
        }
        }
    ]

    extract = NexusFunctionModuleExtractor(class_test, args)
    dummy_test = ['test1','test2']
    x_new, y_new = extract(dummy_test, ['y1','y2'])
    assert type(x_new) == scipy.sparse.csr.csr_matrix
