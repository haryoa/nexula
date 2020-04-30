"""
THis module solely focused on using pytorch lightning on millenial nn
EVERY NN will be stored in model_millenial_helper

For version 0.1
CNN + LSTM
"""
import logging

import torch
from pytorch_lightning import Trainer

from nexula.nexula_data.data_loader import NexusMillenialDataLoader
from nexula.nexula_inventory.inventory_base import NexusBaseModelInventory
from nexula.nexula_inventory.inventory_model.model_millenial_helper.helper_millenial_lightning \
    import LightningModelClassification

logger = logging.getLogger('nexula')


class MillenialBaseClassification(NexusBaseModelInventory):
    """
    Abstract class for Millenial
    """

    type = 'millenial'

    def __init__(self, args, model_name='lstm'):
        super().__init__(args)

        # TODO fix pretrained_model
        self.type = 'millenial'
        if 'additional_info' in args:
            self.preprocess_model_arguments(args)
        logger.debug(self.model_args)

        self.model = LightningModelClassification(num_label=args['additional_info']['num_label'],
                                                  pretrained_vector=args['additional_info']['pretrained_vector'],
                                                  model_params=self.model_args, model_name=model_name)

        # TODO construct callbacks
        self.callbacks = []

        # TODO Check and fix later on
        log_model = self.fit_args['default_save_path'] if \
            'default_save_path' in self.fit_args else \
            'train_ckpt/{}'.format(self.model.__class__.__name__)
        self.log_model = log_model
        self.trainer = Trainer(**self.fit_args, default_save_path=log_model, callbacks=self.callbacks,
                               auto_lr_find=True)

    def preprocess_model_arguments(self, args):
        """
        Preprocessing argument for the model arguments which goes into model.
        This function will handle handling inputting word embedding model.
        """
        # REFACTOR
        logger.debug(args)
        label_dist = args['additional_info']['label_dist']
        text_field = args['additional_info']['text_field']
        max_label = len(label_dist)
        num_label = max_label if max_label > 2 else 1  # num label == 2, it is binary classification
        args['additional_info']['num_label'] = num_label

        word_embedding_file = self.model_args['pretrained_vector'] if 'pretrained_vector' in self.model_args else None
        if word_embedding_file is not None:
            logger.info("PRETRAINED MODEL LOADED")
            vectors = self._read_word_embeddings(word_embedding_file)
            text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
            args['additional_info']['pretrained_vector'] = text_field.vocab.vectors
            # Update embedding_dim
            self.model_args['embedding_dim'] = vectors.dim
            self.model_args.pop('pretrained_vector', None)
        else:
            args['additional_info']['pretrained_vector'] = text_field.vocab.vectors
        self.model_args['num_embedding'] = len(text_field.vocab.stoi)
        logger.info(args)

    def _read_word_embeddings(self, file):
        from torchtext.vocab import Vectors
        from pathlib import Path
        path = Path(file)
        vectors = Vectors(name=path.name, cache=path.parent)
        return vectors

    def extend_callbacks(self, lightning_callback_object):
        """
        Add list of callbacks into lightning

        Parameters
        ----------
        lightning_callback_object :
            list of callbacks object

        Returns
        -------

        """
        self.callbacks.extend(lightning_callback_object)
        # CHECK IF ERROR
        self.trainer = Trainer(**self.fit_args, default_save_path=self.log_model, callbacks=self.callbacks,
                               auto_lr_find=True)

    def fit_to_dataloader(self, data_loader, *args, **kwargs):
        if 'val_dataloader' not in kwargs:
            raise Exception("Need validation dataloader")
        if 'test_dataloader' not in kwargs:
            raise Exception("Need test dataloader")
        val_dataloader = kwargs['val_dataloader']
        test_dataloader = kwargs['test_dataloader']
        logger.debug('callback, {}'.format(self.callbacks))
        trainer = self.trainer

        trainer.fit(self.model, data_loader.get_all_processed_data(fit_to_data=False)[0],
                    val_dataloader.get_all_processed_data()[0],
                    test_dataloader.get_all_processed_data()[0])

    def fit(self, x_train, y_train, *args, **kwargs):
        pass

    def predict(self, x: torch.Tensor, *args, **kwargs):
        import numpy as np
        pred = self.predict_proba(x)
        if self.model.num_label == 1:  # Binary Classification
            pred = np.round(pred, 0)
            pred = pred.ravel()
            pred = pred.astype(np.int)
        else:  # Multi Classification
            pred = np.argmax(pred, 1)
        return pred

    def predict_from_dataloader(self, data_loader: NexusMillenialDataLoader, use_gpu=False, *args, **kwargs):
        """
        Predict from dataloader
        """
        import numpy as np
        b_iterator = data_loader.get_all_processed_data(fit_to_data=False)[0]
        collected_result = np.array([], dtype=np.int)
        for data in b_iterator:
            x, y = data.x, data.y
            pred = self.predict(x)
            collected_result = np.concatenate([collected_result, pred])
        return collected_result

    def prepare_inference(self, use_gpu=False):
        """
        prepare inference such as model goes to eval mode.
        """
        if not use_gpu:
            self.model.cpu()
        else:
            self.model.cuda()
        self.model.eval()
        self.model.freeze()

    def predict_proba(self, x, use_gpu=False, *args, **kwargs):
        """
        Predict with confidence level score / probability
        """
        self.prepare_inference(use_gpu)
        return self.model.predict_proba(x).detach().numpy()

    def predict_proba_from_data_loader(self, data_loader, *args, **kwargs):
        pass


class MillenialCNN1DClassification(MillenialBaseClassification):

    def __init__(self, model_args):
        super().__init__(model_args, 'cnn1d')


class MillenialLSTMClassification(MillenialBaseClassification):

    def __init__(self, model_args):
        super().__init__(model_args, 'lstm')
