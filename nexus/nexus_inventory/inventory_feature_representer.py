import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nexus.nexus_inventory.inventory_base import NexusBaseDataInventory


class NexusTfIdfSklearnRepresenterData(NexusBaseDataInventory):

    def __init__(self, **kwargs):
        self.model = []

    def get_model(self):
        # TODO Wrap model
        return self.model[0]

    def _fit_tfidf(self, x, tf_idf_args, **kwargs):
        return TfidfVectorizer(tf_idf_args).fit(x)

    def __call__(self, x, y, fit_to_data=False, *args, **kwargs):
        """
        Use TF IDF Vectorizer to represent the data as the input of the Machine Learning Model.
        IF the tf-idf is not fitted, the tf-idf model will be fitted automatically.

        Add 'args_TfidfVectorizer' to add argument to the TfIdfVectorizer.
        NOTE that the TFIdfvectorizer use sklearn TfIdfvectorizer, so the arguments should follow it.

        Parameters
        ----------
        x
        y
        fit_to_data
        args
        kwargs

        Returns
        -------

        """
        if 'args_TfidfVectorizer' not in kwargs:
            kwargs['args_TfidfVectorizer'] = {}
        if len(self.model) == 0:
            """ 
            Will always fit to tf idf if no model detected
            """
            vectorizer = self._fit_tfidf(x, kwargs['args_TfidfVectorizer'])
            self.model.append(vectorizer)
            x = self.model[0].transform(x)
        else:
            if fit_to_data:
                vectorizer = self._fit_tfidf(x, kwargs['args_TfidfVectorizer'])
                self.model[0] = vectorizer
            else:
                vectorizer = self.model[0]
            x = vectorizer.transform(x)
        return x, y
