from nexus.nexus_data.data_reader import read_csv
from functools import partial
from nexus.nexus_inventory.inventory_translator import NEXUS_INVENTORY_DATA_READER as nexus_inv_data_reader
from nexus.nexus_utility.utility_extract_func import NexusFunctionModuleExtractor
from nexus.nexus_utility.utility_import_var import import_class
from nexus.nexus_data.data_loader import NexusBoomerDataLoader
# TODO next version, documenting the parameters


class DataController:

    def __init__(self, data_choice_type, **kwargs):
        """

        """
        self.data_choice_type = data_choice_type

    def construct_data_loader(self, data_reader_type, data_reader_args, data_preprocesser_func_list_and_args=[],
                              data_representer_func_list_and_args=[], data_postprocesser_proc=[], **kwargs):
        """
        Construct data loader based on the data_choice_type.
        If manual_split is used, this function will return train, dev, test dataloader

        Parameters
        ----------
        data_reader_type
        data_reader_args
        data_preprocesser_func_list_and_args
        data_representer_func_list_and_args
        data_postprocesser_proc
        kwargs

        Returns
        -------

        """
        if self.data_choice_type == 'manual_split':
            """
            Data reader must contains : 
            {
                'train' : data_reader_args
                'dev' : data_reader_args
                'test' : data_reader_args
            }
            """
            reader_func = import_class(*(nexus_inv_data_reader[data_reader_type]))

            data_reader_func_train = partial(reader_func, **data_reader_args['train'])

            # USE DEFAULT TODO : CHANGE INTO MODULE TOO
            data_preprocesser = self._get_nexus_mod_extractor(data_preprocesser_func_list_and_args)
            data_representer = self._get_nexus_mod_extractor(data_representer_func_list_and_args)
            data_post_preprocess = self._get_nexus_mod_extractor(data_postprocesser_proc)

            # CALL DATA LOADER FOR EACH SET
            data_loader_train = NexusBoomerDataLoader(data_reader_func_train, data_preprocesser,
                                                      data_representer, data_post_preprocess)

            # FOR VALIDATION AND TEST, DO NOT FIT PREPROCESSOR, REPRESENTER, AND POSTPROCESSOR
            data_reader_func_dev = partial(reader_func, **data_reader_args['dev'])
            data_loader_dev = NexusBoomerDataLoader(data_reader_func_dev, data_preprocesser,
                                                    data_representer, data_post_preprocess)

            # FOR VALIDATION AND TEST, DO NOT FIT PREPROCESSOR, REPRESENTER, AND POSTPROCESSOR
            data_reader_func_test = partial(reader_func, **data_reader_args['test'])
            data_loader_test = NexusBoomerDataLoader(data_reader_func_test, data_preprocesser,
                                                    data_representer, data_post_preprocess)
            return data_loader_train, data_loader_dev, data_loader_test

    def _get_nexus_mod_extractor(self, list_func_args, **kwargs):
        """
        Instantiate ModuleExtractor which instantiate and the objec
        Parameters
        ----------
        list_func_args
        kwargs

        Returns
        -------

        """
        proc = [proc['process'] for proc in list_func_args]
        args = [proc['params'] for proc in list_func_args]
        extractor = NexusFunctionModuleExtractor(proc, args, **kwargs)
        return extractor
