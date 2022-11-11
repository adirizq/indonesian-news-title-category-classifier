import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from utils.preprocessor import NewsDataModule
from models.cnn import CNN1D

if __name__ == '__main__':
    pl.seed_everything(99, workers=True)

    data_module = NewsDataModule(batch_size=128)
    weigths = data_module.word_embedding()

    model = CNN1D(
        word_embedding_weigth=weigths,
    )

    logger = TensorBoardLogger("logs", name="cnn_classifier")
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/cnn', save_last=True, monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)
    tqdm_progress_bar = TQDMProgressBar()

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/cnn",
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module, ckpt_path='best')
