import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\'\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    return text

def load_data(topics_path, opinions_path, conclusions_path):
    topics = pd.read_csv(topics_path)
    opinions = pd.read_csv(opinions_path)
    conclusions = pd.read_csv(conclusions_path)
    return topics, opinions, conclusions

def preprocess_data():
    topics, opinions, conclusions = load_data('data/topics.csv', 'data/opinions.csv', 'data/conclusions.csv')
    topics['text'] = topics['text'].apply(clean_text)
    opinions['text'] = opinions['text'].apply(clean_text)
    conclusions['text'] = conclusions['text'].apply(clean_text)
    return topics, opinions, conclusions

def split_data(opinions: pd.DataFrame, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    splits = splitter.split(opinions, groups=opinions['topic_id'])
    train_idx, test_idx = next(splits)
    train = opinions.iloc[train_idx].reset_index(drop=True)
    test = opinions.iloc[test_idx].reset_index(drop=True)
    return train, test

def split_data_all(opinions: pd.DataFrame, val_size=0.1, test_size=0.1, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    splits = gss.split(opinions, groups=opinions['topic_id'])
    train_temp_idx, test_idx = next(splits)
    train_temp = opinions.iloc[train_temp_idx].reset_index(drop=True)
    test = opinions.iloc[test_idx].reset_index(drop=True)

    val_relative_size = val_size / (1 - test_size)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
    splits = gss.split(train_temp, groups=train_temp['topic_id'])
    train_idx, val_idx = next(splits)
    train = train_temp.iloc[train_idx].reset_index(drop=True)
    val = train_temp.iloc[val_idx].reset_index(drop=True)

    return train, val, test
