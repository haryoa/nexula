# TODO next version, documenting the parameters
import logging
from functools import partial
from typing import List

from nexula.nexula_data.data_loader import NexusBoomerDataLoader, NexusMillenialDataLoader, BaseNexusDataLoader
from nexula.nexula_inventory.inventory_translator import NEXUS_INVENTORIES
from nexula.nexula_utility.utility_extract_func import NexusFunctionModuleExtractor
from nexula.nexula_utility.utility_import_var import import_class

logger = logging.getLogger('nexula')


class DataController:

    def __init__(self, data_choice_type: str, model_type: str = 'boomer', **kwargs):
        """
        Data choice type how to manage the data.
        The current only option is 'manual_split'
        Parameters
        ----------
        data_choice_type: str
            Type on managing the data
        model_type: str
            Model type (note that millenial and boomer model type will be different)
        """
        self.data_choice_type = data_choice_type
        self.model_type = model_type
        self.dataloader_obj = NexusBoomerDataLoader if model_type == 'boomer' else NexusMillenialDataLoader

    def construct_data_loader(self,
                              data_reader_type: str, data_reader_args: dict,
                              data_preprocesser_func_list_and_args: List[dict] = None,
                              data_representer_func_list_and_args=None,
                              data_postprocesser_proc=None, **kwargs) -> (BaseNexusDataLoader,
                                                                          BaseNexusDataLoader,
                                                                          BaseNexusDataLoader):
        """
        Construct data loader based on the data_choice_type.
        If manual_split is used, this function will return train, dev, test dataloader

        Parameters
        ----------
        data_reader_type: str
            The data reader. This string needs to follow nexula_inventory translator.
        data_reader_args: dict
            The data reader arguments when invoke the function.
        data_preprocesser_func_list_and_args: List[dict]
            List of preprocesser function
        data_representer_func_list_and_args: List[dict]
            List of representer function. The last process in the pipeline MUST return
            the data which is ready to be input of the model
        data_postprocesser_proc: List[dict]
            Any post process after changing data representation
        Returns
        -------
        BaseNexusDataLoader, BaseNexusDataLoader, BaseNexusDataLoader
            Return data loader. if 'manual_split' is chosen, it will return 3 dataloader
            which represent train, dev, and test
        """

        if data_representer_func_list_and_args is None:
            data_representer_func_list_and_args = []
        if data_preprocesser_func_list_and_args is None:
            data_preprocesser_func_list_and_args = []
        if data_postprocesser_proc is None:
            data_postprocesser_proc = []
        data_loader_train, data_loader_dev, data_loader_test = None, None, None
        if self.data_choice_type == 'manual_split':
            # Need to be refactored
            reader_func = import_class(*(NEXUS_INVENTORIES[data_reader_type]))

            data_reader_func_train = partial(reader_func, **data_reader_args['train'])
            data_preprocesser = self._get_nexus_mod_extractor(data_preprocesser_func_list_and_args)
            data_representer = self._get_nexus_mod_extractor(data_representer_func_list_and_args)
            data_post_preprocess = self._get_nexus_mod_extractor(data_postprocesser_proc)

            # create all dataloaders for every dataset , note that this needs to be refactored
            data_loader_train = self.dataloader_obj(data_reader_func_train, data_preprocesser,
                                                    data_representer, data_post_preprocess)

            data_reader_func_dev = partial(reader_func, **data_reader_args['dev'])
            data_loader_dev = self.dataloader_obj(data_reader_func_dev, data_preprocesser,
                                                  data_representer, data_post_preprocess)

            data_reader_func_test = partial(reader_func, **data_reader_args['test'])
            data_loader_test = self.dataloader_obj(data_reader_func_test, data_preprocesser,
                                                   data_representer, data_post_preprocess)
            logger.info("Done on creating train, dev, and test dataloader")
        return data_loader_train, data_loader_dev, data_loader_test

    def _get_nexus_mod_extractor(self, list_func_args, **kwargs):
        """
        Create a builder

        Parameters
        ----------
        list_func_args
        kwargs

        Returns
        -------

        """
        proc = [proc['process'] for proc in list_func_args]
        args = [proc['params'] for proc in list_func_args]
        logger.debug(proc)
        extractor = NexusFunctionModuleExtractor(proc, args, **kwargs)
        return extractor
