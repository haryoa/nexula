import logging

import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from torchtext.data import Field, BucketIterator, Iterator

from nexula.nexula_inventory.inventory_base import NexusBaseDataInventory
from nexula.nexula_inventory.inventory_translator import NEXUS_INVENTORY_DATA_READER as nexus_inv_data_reader
from nexula.nexula_utility.utility_import_var import import_class

logger = logging.getLogger("nexula.representer")


class NexusTfIdfSklearnRepresenterData(NexusBaseDataInventory):

    def __init__(self, **kwargs):
        self.model = []

    def get_model(self):
        # TODO Wrap model
        return self.model[0]

    def _fit_tfidf(self, x, tf_idf_args, **kwargs):
        logger.info("Fit TF IDF")
        return TfidfVectorizer(tf_idf_args).fit(x)

    def __call__(self, x, y, fit_to_data=False, *args, **kwargs):
        """
        Use TF IDF Vectorizer to represent the data as the input of the Machine Learning Model.
        IF the tf-idf is not fitted, the tf-idf model will be fitted automatically.

        Add 'args_TfidfVectorizer' to add argument to the TfIdfVectorizer.
        NOTE that the TFIdfvectorizer use sklearn TfIdfvectorizer, so the arguments should follow it.

        Returns
        -------
        np.array, np.array
            Return preprocessed input and output label respectively

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


class NexusMillenialTorchTextRepresenter(NexusBaseDataInventory):
    """
    This module will be used to represent feature for deep learning in Pytorch
    """

    def __init__(self, tokenizer_name='nltk_wordtokenize', seq_len: int = 50, lowercase: bool = True,
                 batch_size: int = 64, vocab_size: int = None, min_freq: int = 1, fit_first: str = 'fit_to_train',
                 fit_first_args=None, fit_first_custom_data=None, binary=False):
        """
        Torch Text iterator/dataloader creator
        Parameters
        ----------
        tokenizer_name: str
            How to split the text. Usage = 'nltk_wordtokenize', 'default','spacy'
        seq_len: int
            Max sequence length for the input of the model
        lowercase: bool
            Whether to lowercase the text or not
        batch_size: int
            Batch size on loading the data
        vocab_size: int
            Max Vocabulary size. None if unlimited.
        min_freq: int
            Minimum frequency of a token to be added to vocabulary
        fit_first: str
            Fitting strategy. None if fitting manually on training set
        fit_first_args: dict
            Fitting arguments strategy.
            'manual_split' strategy needs data_choice_type, data_reader_type
            data_reader_args
        fit_first_custom_data: str
            Other data NOT IMPLEMENTED YET
        binary: bool
            To change the label type.
            torch.Long if multiclass
            torch.float if binary
        """
        import torch
        dtype = torch.float if binary else torch.long
        self.logger.info("Tokenizing data with {}".format(tokenizer_name))
        self.tokenizer = self.get_tokenizer(tokenizer_name)
        self.models = []
        self.seq_len = seq_len
        self.text_field = Field(sequential=True, tokenize=self.tokenizer, lower=lowercase,
                                init_token='<START>', eos_token='<END>', fix_length=seq_len)
        self.label_field = Field(sequential=False, use_vocab=False, dtype=dtype)
        self.batch_size = batch_size
        self.already_fit = False
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.fit_first(fit_first, fit_first_args, fit_first_custom_data)
        self.label_dist = []

    def fit_first(self, fit_to: str = 'fit_to_train', fit_first_args=None, fit_first_custom_data=None):
        """
        Fit the data by selecting variety of choice.

        fit_first_args arguments -> {
            'data_choice_type' : 'manual_split,
            'data_reader_args' : {},
            'data_reader_type' :
        }
        """
        if fit_to == 'fit_all_dataset':
            if fit_first_args['data_choice_type'] == 'manual_split':
                data_reader_type = fit_first_args['data_reader_type']
                data_reader_args = fit_first_args['data_reader_args']
                reader_func = import_class(*(nexus_inv_data_reader[data_reader_type]))
                x_train, _ = reader_func(**data_reader_args['train'])
                x_dev, _ = reader_func(**data_reader_args['dev'])
                x_test, _ = reader_func(**data_reader_args['test'])
                x_combined = np.concatenate([x_train, x_dev, x_test])
                logger.info("Fitting Vocabulary (Torch Text) with option {}".format(fit_to))
                self.fit(x_combined)
            else:
                raise NotImplementedError('{} is not implemented yet'.format(fit_first_args['data_choice_type']))

    def get_tokenizer(self, tokenizer_name='nltk_wordtokenize'):
        if tokenizer_name == 'nltk_wordtokenize':
            return word_tokenize
        elif tokenizer_name == 'default':
            return lambda x: x.split()
        elif tokenizer_name == 'spacy':
            return 'spacy'
        else:
            raise NotImplementedError("Tokenizer {} is not implemented".format(tokenizer_name))

    def get_model(self):
        return self.text_field

    def _transform(self, x):
        pass

    def fit(self, x):
        """

        Parameters
        ----------
        x: list[str]
            List of string

        Returns
        -------

        """
        x = [self.text_field.preprocess(x_elem) for x_elem in x]
        self.already_fit = True
        self.text_field.build_vocab(x, max_size=self.vocab_size, min_freq=self.min_freq)

    def __call__(self, x, y, fit_to_data=False, shuffle=True, *args, **kwargs):
        # REFACTOR THIS
        from ..nexula_inventory.representer_torchtext.torchtext_helper import DataFrameDataset

        if not self.already_fit:
            self.fit(x)

        df = pd.DataFrame(dict(x=x, y=y))
        if fit_to_data:
            self.label_dist = df.y.value_counts().to_dict()
        ds = DataFrameDataset(df, dict(x=self.text_field, y=self.label_field))
        if fit_to_data:
            iterate = BucketIterator(dataset=ds, batch_size=self.batch_size, sort_key=lambda x: len(x.text),
                                     shuffle=True)
        else:
            iterate = Iterator(dataset=ds, batch_size=self.batch_size, shuffle=False)
        return iterate, y

    def _tokenize(self, texts: np.array):
        self.tokenizer.fit_on_texts()
