import pytorch_lightning as pl
import re
import nltk
import string
import torch
import pickle

from nltk.corpus import stopwords
from keras.utils import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from models.lstm import LSTM


class Prediction():

    def __init__(self):
        pl.seed_everything(99, workers=True)

        self.labels = ['Finance', 'Food', 'Health', 'Hot', 'Inet', 'News', 'Oto', 'Sport', 'Travel']

        with open('models/w2v_matrix.pkl', 'rb') as handle:
            embedding_matrix = pickle.load(handle)

        with open('utils/tokenizer.pkl', 'rb') as handle:
            self.max_len, self.tokenizer = pickle.load(handle)

        self.model = LSTM.load_from_checkpoint('checkpoints/lstm/epoch=18-step=9728.ckpt', word_embedding_weigth=embedding_matrix)
        self.model.eval()

        nltk.download('stopwords')
        self.stop_words = stopwords.words('indonesian')

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def predict(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        text = ' '.join([item for item in text.split() if item not in self.stop_words])

        text = self.stemmer.stem(text)

        text = [text]
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=self.max_len, padding='pre')

        text = torch.tensor(text)
        pred = torch.argmax(self.model(text), dim=1)

        return self.labels[pred]
