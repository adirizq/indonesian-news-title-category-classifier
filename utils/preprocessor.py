import sys
import torch
import os
import nltk
import re
import string
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import multiprocessing
import pickle

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import word2vec
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm


nltk.download('punkt')


class NewsDataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset_path='datasets/indonesian-news-title.csv',
                 w2v_model_path='models/w2v/idwiki_word2vec_200_new_lower.model',
                 preprocessed_data_path='datasets/preprocessed/data.pkl',
                 tokenizer_path='utils/tokenizer.pkl',
                 embedding_matrix_path='models/w2v_matrix.pkl',
                 batch_size=64,):

        super(NewsDataModule, self).__init__()

        self.dataset_path = dataset_path
        self.w2v_model_path = w2v_model_path
        self.preprocessed_data_path = preprocessed_data_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.embedding_matrix_path = embedding_matrix_path

        if os.path.exists(preprocessed_data_path):
            print("\nLoading Preprocessed Data...")
            self.dataset = pd.read_pickle(preprocessed_data_path)
            print('[Loading Completed]\n')
        else:
            print("\nPreprocessing Data...")
            self.dataset = self.preprocess()
            print('[Preprocessing Completed]\n')

    def preprocess(self):
        print("\nLoading Data...")
        dataset = pd.read_csv(self.dataset_path)
        dataset = dataset[['title', 'category']]
        print('[Loading Completed]\n')

        dataset = dataset.dropna()

        print("\nIndexing Label...")
        labels = dataset['category'].unique().tolist()
        dataset['category'] = dataset['category'].map(lambda x: labels.index(x))
        print('[Indexing Completed]\n')

        print("\nCleaning Data...")
        tqdm.pandas(desc='Cleaning')
        dataset['title'] = dataset['title'].progress_apply(lambda x: self.clean(x))
        print('[Cleaning Completed]\n')

        print("\nRemoving Stopwords...")
        nltk.download('stopwords')
        stop_words = stopwords.words('indonesian')
        tqdm.pandas(desc='Stopwords Removal')
        dataset['title'] = dataset['title'].progress_apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
        print("[Stopwords Removal Completed]")

        print("\nStemming Data...")
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        tqdm.pandas(desc='Stemming')
        dataset['title'] = dataset['title'].progress_apply(lambda x: stemmer.stem(x))
        print("[Stemming Completed]")

        dataset = dataset.dropna()

        print("\nSaving Preprocessed Dataset...")
        dataset.to_pickle(self.preprocessed_data_path)
        print("[Saving Completed]")

        return dataset

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.split()
        text = ' '.join(text)

        return text.strip()

    def load_data(self):
        x = self.dataset['title']
        y = self.dataset['category']

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)
        list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]
        max_len = max(len(x) for x in list_set_sequence)

        print('Padding Data...')
        x = pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))], maxlen=max_len, padding='pre')
        print('[Padding Completed]\n')

        x = torch.tensor(x)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x, y)

        print('Splitting Data...')
        train_valid_len = round(len(tensor_dataset) * 0.8)
        test_len = len(tensor_dataset) - train_valid_len

        train_valid_data, test_data = torch.utils.data.random_split(
            tensor_dataset, [
                train_valid_len, test_len
            ]
        )

        train_len = round(len(train_valid_data) * 0.9)
        valid_len = len(train_valid_data) - train_len

        train_data, valid_data = torch.utils.data.random_split(
            train_valid_data, [
                train_len, valid_len
            ]
        )
        print('[Splitting Completed]\n')

        return train_data, valid_data, test_data

    def word_embedding(self):
        if os.path.exists(self.embedding_matrix_path):
            with open(self.embedding_matrix_path, 'rb') as handle:
                embedding_matrix = pickle.load(handle)

            return embedding_matrix
        else:
            print('Loading Word2Vec Model...')
            w2v_model = word2vec.Word2Vec.load(self.w2v_model_path)
            w2v_weights = w2v_model.wv
            print('[Loading Completed]\n')

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(self.dataset['title'])

            jumlah_index = len(tokenizer.word_index) + 1

            embedding_matrix = np.zeros((jumlah_index, w2v_weights.vectors.shape[1]))
            print('Creating Embedding Matrix...')
            for word, i in tqdm(tokenizer.word_index.items(), desc='Creating W2V Weigth'):
                try:
                    embedding_vector = w2v_weights[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 200)
            print('[Embedding Matrix Completed]\n')

            del (w2v_weights)

            with open(self.embedding_matrix_path, 'wb') as handle:
                pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return embedding_matrix

    def tokenizer(self):
        if os.path.exists(self.tokenizer_path):
            with open(self.tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)

            sequences = tokenizer.texts_to_sequences(self.dataset['title'])
            list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]
            max_len = max(len(x) for x in list_set_sequence)

            return max_len, tokenizer
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(self.dataset['title'])

            with open(self.tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            sequences = tokenizer.texts_to_sequences(self.dataset['title'])
            list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]
            max_len = max(len(x) for x in list_set_sequence)

            return max_len, tokenizer

    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
