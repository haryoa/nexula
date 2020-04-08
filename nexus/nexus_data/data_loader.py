from nexus.nexus_data.data_reader import read_csv
from nexus.nexus_utility.utility_extract_func import NexusFunctionModuleExtractor


class BaseNexusDataLoader:

    def __init__(self, nexus_data_preprocesser=None, nexus_feature_representer=None,
                 nexus_feature_postprocess=None, **kwargs):
        self.nexus_data_preporcesser = nexus_data_preprocesser
        self.nexus_feature_representer = nexus_feature_representer
        self.nexus_feature_post_process = nexus_feature_postprocess

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

    def __init__(self, nexus_data_reader_func=None, nexus_data_preprocesser: NexusFunctionModuleExtractor=None,
                 nexus_feature_representer: NexusFunctionModuleExtractor=None,
                 nexus_feature_postprocess: NexusFunctionModuleExtractor=None, **kwargs):
        super(NexusBoomerDataLoader, self).__init__(nexus_data_preprocesser,
                                                    nexus_feature_representer,
                                                    nexus_feature_postprocess)
        self.nexus_data_reader_func = nexus_data_reader_func
        self.nexus_data_preprocesser = nexus_data_preprocesser
        self.nexus_feature_representer = nexus_feature_representer
        self.nexus_feature_postprocess = nexus_feature_postprocess

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

    def __len__(self):
        # Num of data
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def get_all_processed_data(self, fit_to_data=True):
        pass
