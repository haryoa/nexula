import importlib

from nexus.nexus_utility.utility_import_var import import_class


class NexusFunctionModuleExtractor():
    """
    Used for constructing pipeline data preporcessing and feature representer
    """

    def __init__(self, module_class_list, args_dict, **kwargs):
        """
        Instantiate object of the classes in the pipeline
        Parameters
        ----------
        module_class_list
        args_dict
        kwargs
        """
        # self.list_of_cls = self._search_module_function(module_class_list)
        self.list_of_cls = module_class_list

        if 'logger' in kwargs:
            self.logger = kwargs['logger']

        self.logger.debug(args_dict) if 'logger' in self.__dict__ else None
        self.args_init = [arg['init'] for arg in args_dict]
        self.args_call = [arg['call'] for arg in args_dict]
        self._construct_object()
        # Extract call

    def _construct_object(self):
        """
        Instantiate object of all pipeline
        """
        new_list_of_cls = []
        for i, cls in enumerate(self.list_of_cls): #REFACTOR
            new_list_of_cls.append(cls(**self.args_init[i]))
        self.list_of_cls = new_list_of_cls

    def _search_module_function(self, module_function_list):
        """
        Search the module in the library
        Parameters
        ----------
        module_function_list

        Returns
        -------

        """
        list_of_cls = []
        for module, function in module_function_list:
            # TODO Raise exception if empty
            list_of_cls.append(import_class(function, module))
        return list_of_cls

    def __call__(self, x, y, *args, **kwargs):
        """
        Call the object by evoking __call__ function

        Returns
        -------

        """
        for i,cls in enumerate(self.list_of_cls):
            current_args = self.args_call[i]
            x, y = cls(x, y, **kwargs, **current_args)
        return x, y
