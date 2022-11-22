# Indonesian News Title Category CLassifier
> This is my final project for Information Retrieval Course

Want to give it a try? Try it now on [![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indonesian-news-title-category-classifier.streamlit.app/)

## Dataset
[Indonesian News Title](https://www.kaggle.com/datasets/ibamibrahim/indonesian-news-title) by Ibrahim on Kaggle. This dataset contains **91,017** Indonesian News Title caregorized into **9** labels ('finance', 'food', 'health', 'hot', 'inet', 'news', 'oto', 'sport', 'travel').

### Example
| Date  | Url | Title | Category |
| --- | --- | --- | --- |
| 02/26/2020  | https://finance.detik.com/berita-ekonomi-bisnis/d-4916114/kemnaker-awasi-tka-di-meikarta  | Kemnaker Awasi TKA di Meikarta  | finance  |
| 02/27/2020  | https://sport.detik.com/detiktv/d-4916359/kangen-rooney-lihat-nih-gol-gol-kerennya  | Kangen Rooney? Lihat Nih Gol-gol Kerennya  | sport |
| 04/22/2020  | https://travel.detik.com/travel-news/d-4985787/jadikan-rumah-serasa-tempat-travelling-ini-tipsnya  | Jadikan Rumah Serasa Tempat Travelling, Ini Tipsnya  | travel  |

## Result
| Model  | Test Accuracy | Train Loss | Validation Loss |
| --- | --- | --- | --- |
| [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) + [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) | 89.16% | 0.4327 | 0.4741 |
| [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) + [CNN 1D](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) | 88.53% | 0.4324 | 0.5803 |

## Project Structure
```bash
├── datasets
│   ├── preprocessed
│   │   ├── data.pkl
│   └── indonesian-news-title.csv
├── models
│   ├── cnn.py
│   ├── lstm.py
│   ├── inference.py
│   └── w2v_matrix.pkl
├── utils
│   ├── preprocessor.py
│   └── tokenizer.pkl
├── app.py
├── train_cnn.py
├── train_lstm.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Streamlit Demo
```bash
streamlit run app.py
```

## Author
[Rizky Adi](https://www.linkedin.com/in/rizky-adi-7b008920b/)


Last update: 22 November 2022
