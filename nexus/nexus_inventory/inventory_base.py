import pandas as pd


class NexusBaseDataInventory:

    def __init__(self, **kwargs):
        self.model = []

    def get_model(self):
        # TODO Wrap model
        raise NotImplementedError

    def __call__(self, x, y, *args, **kwargs):
        raise NotImplementedError


class NexusBaseModelInventory:

    def  __init__(self, model_args):
        # TODO add resume training
        self.type = 'base'
        self.model_args = model_args['init']
        self.fit_args = model_args['call']

    def fit_to_dataloader(self, data_loader, *args, **kwargs):
        """
        Fit the data to the model when using nexus dataloader.
        Parameters
        ----------
        data_loader
        args
        kwargs

        Returns
        -------

        """
        raise NotImplementedError

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fit the model with raw variable
        Parameters
        ----------
        x_train
        y_train
        args
        kwargs

        Returns
        -------

        """
        raise NotImplementedError

    def predict(self, x, *args, **kwargs):
        """
        Predict x by using model
        Parameters
        ----------
        x
        args
        kwargs

        Returns
        -------

        """
        raise NotImplementedError

    def predict_from_dataloader(self, data_loader, *args, **kwargs):
        """
        Predict by using dataloader (nexus)
        Parameters
        ----------
        data_loader
        args
        kwargs

        Returns
        -------

        """

        raise NotImplementedError

    def predict_proba(self, x, *args, **kwargs):
        """
        Predict and output the probability using raw data
        Parameters
        ----------
        x
        args
        kwargs

        Returns
        -------

        """
        pass

    def predict_proba_from_data_loader(self, data_loader, *args, **kwargs):
        """
        Predict and output the probability usingn dataloader (nexus)
        Parameters
        ----------
        data_loader
        args
        kwargs

        Returns
        -------

        """
        pass


class NexusBaseCallbackInventory:

    def before_loop_step(self, trainer_object):
        """
        This function will be called BEFORE looping the model
        """
        pass

    def after_loop_step(self, trainer_object):
        """
        This function will be called AFTER looping the model
        """
        pass

    def in_loop_before_fit_step(self, trainer_object):
        """
        This function will be called INSIDE loop BEFORE fitting the model
        """
        pass

    def in_loop_after_fit_step(self, trainer_object):
        """
        This function will be called INSIDE loop AFTER fitting the model
        """
        pass

