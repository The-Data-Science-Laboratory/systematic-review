# パッケージのインポート
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# csvの相対パス
path = './data/train.csv'
data = pd.read_csv(path)
data['title'] = data['title'].astype(str)
data['abstract'] = data['abstract'].astype(str)


# テキストの前処理関数を定義する
def preprocess_text(text):
    # 小文字化
    text = text.lower()
    # 特殊文字の除去
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    # 単語のトークン化
    tokens = word_tokenize(text)
    # ストップワードの除去
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # 単語のレンマ化
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # 空白を除去してトークンを再結合
    text = ' '.join(tokens)
    return text


# タイトルとアブストラクトの列に前処理を適用する
data['clean_title'] = data['title'].apply(preprocess_text)
data['clean_abstract'] = data['abstract'].apply(preprocess_text)

# 前処理済みデータの最初の数行を表示して確認する
print(data[['clean_title', 'clean_abstract']].head())