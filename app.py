import pytorch_lightning as pl
import re
import nltk
import string
import torch

from flask import Flask, render_template, request
from nltk.corpus import stopwords
from keras.utils import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from utils.preprocessor import NewsDataModule
from models.lstm import LSTM

app = Flask(__name__)


class Prediction():

    def __init__(self):
        pl.seed_everything(99, workers=True)

        self.labels = ['Finance', 'Food', 'Health', 'Hot', 'Inet', 'News', 'Oto', 'Sport', 'Travel']

        data_module = NewsDataModule(batch_size=128)
        self.max_len, self.tokenizer = data_module.tokenizer()
        weigths = data_module.word_embedding()

        self.model = LSTM.load_from_checkpoint('checkpoints/lstm/epoch=18-step=9728.ckpt', word_embedding_weigth=weigths)
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


pred_class = Prediction()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    text = request.form['title']
    pred_result = pred_class.predict(text)

    return render_template('index.html', prediction=pred_result, title=text)


if __name__ == '__main__':
    app.run(debug=True, port=3000)
