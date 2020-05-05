import logging

from nexula.nexula_utility.utility_extract_func import NexusFunctionModuleExtractor
from typing import Iterator

logger = logging.getLogger('nexula')


class BaseNexusDataLoader:

    def __init__(self, nexus_data_preprocesser=None, nexus_feature_representer=None,
                 nexus_feature_postprocess=None, **kwargs):
        self.nexus_data_preporcesser = nexus_data_preprocesser
        self.nexus_feature_representer = nexus_feature_representer
        self.nexus_feature_post_process = nexus_feature_postprocess
        self.label_dist = {}

    def __len__(self):
        # Num of data
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_all_processed_data(self, fit_to_data=True):
        raise NotImplementedError
    # def __iter__(self):
    #     """Create a generator that iterate over the Sequence."""
    #     for item in (self[i] for i in range(len(self))):
    #         yield item


class NexusBoomerDataLoader(BaseNexusDataLoader):

    def __init__(self, nexus_data_reader_func=None, nexus_data_preprocesser: NexusFunctionModuleExtractor = None,
                 nexus_feature_representer: NexusFunctionModuleExtractor = None,
                 nexus_feature_postprocess: NexusFunctionModuleExtractor = None, **kwargs):
        super(NexusBoomerDataLoader, self).__init__(nexus_data_preprocesser,
                                                    nexus_feature_representer,
                                                    nexus_feature_postprocess)
        self.nexus_data_reader_func = nexus_data_reader_func
        self.nexus_data_preprocesser = nexus_data_preprocesser
        self.nexus_feature_representer = nexus_feature_representer
        self.nexus_feature_postprocess = nexus_feature_postprocess
        self.label_dist = self.get_label_dist()

    def get_label_dist(self):
        import collections
        _, y = self.get_all_processed_data(False)
        return collections.Counter(y)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        pass

    def get_all_processed_data(self, fit_to_data=True):
        # Data Preprocessing (args)
        self.x_data, self.y_data = self.nexus_data_reader_func()
        self.x_data, self.y_data = self.nexus_data_preprocesser(self.x_data, self.y_data, fit_to_data=fit_to_data)

        # Feature Representation (args)
        self.x_data, self.y_data = self.nexus_feature_representer(self.x_data, self.y_data, fit_to_data=fit_to_data)

        # Other preprocessing on feature (if needed)
        self.x_data, self.y_data = self.nexus_feature_postprocess(self.x_data, self.y_data, fit_to_data=fit_to_data)

        # Return processed one
        return self.x_data, self.y_data


class NexusMillenialDataLoader(BaseNexusDataLoader):

    def __init__(self, nexus_data_reader_func=None, nexus_data_preprocesser: NexusFunctionModuleExtractor = None,
                 nexus_feature_representer: NexusFunctionModuleExtractor = None, *args, **kwargs):
        super(NexusMillenialDataLoader, self).__init__(nexus_data_preprocesser,
                                                       nexus_feature_representer,
                                                       None)
        self.nexus_data_reader_func = nexus_data_reader_func
        self.nexus_data_preprocesser = nexus_data_preprocesser
        self.nexus_feature_representer = nexus_feature_representer
        self.dataloader = None
        self._x_data = None
        self._y_data = None
        self.label_dist = {}

    def __len__(self):
        return len(self._y_data)

    def __getitem__(self, index):
        pass

    def get_all_processed_data(self, fit_to_data=True) -> Iterator[any]:
        # Data Preprocessing (args)
        self._x_data, self._y_data = self.nexus_data_reader_func()
        self._x_data, self._y_data = self.nexus_data_preprocesser(self._x_data, self._y_data,
                                                                  fit_to_data=fit_to_data)

        # Feature Representation (args)
        self.dataloader, y = self.nexus_feature_representer(self._x_data, self._y_data,
                                                            fit_to_data=fit_to_data)

        import collections
        self.label_dist = collections.Counter(y)
        print(self.label_dist)

        # Return processed one
        return self.dataloader, y
