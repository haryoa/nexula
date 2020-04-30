import logging

import pytorch_lightning as pl
import torch

from nexula.nexula_inventory.inventory_model.architecture_torch import TorchCNN1DClassification
from nexula.nexula_inventory.inventory_model.architecture_torch import TorchLSTMClassification

logger = logging.getLogger('nexula.train')


class LightningModelClassification(pl.LightningModule):

    def __init__(self, model_name, model_params, train_dataloader=None, num_label=1,
                 val_dataloader=None, test_dataloader=None, pretrained_vector=None):
        super().__init__()
        model_params['num_label'] = num_label
        if model_name == 'cnn_1d':
            self.model = TorchCNN1DClassification(pretrained_vector=pretrained_vector, **model_params)
        elif model_name == 'lstm':
            self.model = TorchLSTMClassification(pretrained_vector=pretrained_vector, **model_params)
        else:
            self.model = TorchCNN1DClassification(pretrained_vector=pretrained_vector, **model_params)
        self.num_label = num_label
        self.train_input_dataloader = train_dataloader
        self.val_input_dataloader = val_dataloader
        self.test_input_dataloader = test_dataloader

    def forward(self, x):
        # x =  (seq_len, bs)
        nn_out = self.model(x)
        return nn_out

    def predict_proba(self, x):
        nn_out = self.model(x)
        if self.num_label == 1:
            nn_out = torch.sigmoid(nn_out)
        else:
            nn_out = torch.softmax(nn_out, dim=1)
        return nn_out

    def configure_optimizers(self):
        from torch.optim import Adam
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, 'val')

    def _shared_step(self, batch, batch_idx, prefix='val'):
        import torch
        from torch.nn import functional as F

        x, y = batch.x, batch.y
        if self.trainer.on_gpu:
            x = x.cuda()
            y = y.cuda()
        logits = self(x)

        if self.num_label == 1:
            logits = logits.squeeze(1)
            y_pred = F.sigmoid(logits)
            y_pred = torch.round(y_pred)
            logger.info((y_pred, y))
            loss = torch.nn.BCEWithLogitsLoss()(logits, y)
        else:
            y_pred = F.softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            loss = torch.nn.CrossEntropyLoss()(logits, y)

        # add logging
        prefix = '{}_'.format(prefix) if prefix != 'train' else ''
        if prefix == '':
            return {'{}loss'.format(prefix): loss}
        else:
            return {'{}loss'.format(prefix): loss, 'y_pred': y_pred, 'y_real': y}

    def _shared_epoch_end(self, outputs, prefix='val'):
        print("aaa")
        y_pred = torch.cat([x['y_pred'] for x in outputs], 0)
        y_real = torch.cat([x['y_real'] for x in outputs], 0)
        from nexula.nexula_inventory.inventory_metrics import accuracy_score
        acc_score = accuracy_score(y_real.cpu().numpy(), y_pred.cpu().numpy())
        avg_loss = torch.stack([x['{}_loss'.format(prefix)] for x in outputs]).mean()
        tensorboard_logs = {'{}_loss'.format(prefix): avg_loss, '{}_acc_score'.format(prefix): acc_score}
        return {'{}_loss'.format(prefix): avg_loss,
                'progress_bar': tensorboard_logs,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, 'test')
