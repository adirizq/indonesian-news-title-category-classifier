import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class CNN1D(pl.LightningModule):
    def __init__(self,
                 word_embedding_weigth,
                 out_channels=128,
                 window_sizes=[3, 4, 5],
                 num_classes=9,
                 learning_rate=1e-3,
                 dropout=0.5,
                 ) -> None:

        super(CNN1D, self).__init__()

        self.lr = learning_rate
        self.output_dim = num_classes
        self.out_channels = out_channels

        weights = torch.FloatTensor(word_embedding_weigth)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=word_embedding_weigth.shape[1], out_channels=out_channels, kernel_size=window_size) for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((out_channels*len(window_sizes)), num_classes)
        self.tanh = nn.Tanh()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        embedding_out = self.embedding(x)

        prepared_conv_input = embedding_out.permute(0, 2, 1)

        out_conv = []

        for conv in self.conv1d:
            x = conv(prepared_conv_input)
            x = self.tanh(x)
            x = F.max_pool1d(x, x.size(2))
            out_conv.append(x)

        logits = torch.cat(out_conv, 1)
        logits = logits.squeeze(-1)

        logits = self.dropout(logits)
        logits = self.linear(logits)

        return logits

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
