import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class LSTM(pl.LightningModule):
    def __init__(self,
                 word_embedding_weigth,
                 hidden_size=128,
                 num_classes=9,
                 learning_rate=1e-3,
                 num_layers=3,
                 dropout=0.5,
                 ) -> None:

        super(LSTM, self).__init__()

        self.lr = learning_rate
        self.output_dim = num_classes
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        weights = torch.FloatTensor(word_embedding_weigth)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(input_size=word_embedding_weigth.shape[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        embedding_out = self.embedding(x)
        embedding_out = self.dropout(embedding_out)
        lstm_out, _ = self.lstm(embedding_out)
        dropout_out = self.dropout(lstm_out)
        out = dropout_out[:, -1, :]
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        titles, targets = train_batch

        out = self(titles)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'train_loss': loss, 'train_acc': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        titles, targets = valid_batch

        out = self(titles)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        titles, targets = test_batch

        out = self(titles)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'test_loss': loss, 'test_acc': accuracy}, prog_bar=True, on_epoch=True)

        return loss
