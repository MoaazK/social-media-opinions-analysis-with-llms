import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_models():
    # Load grouping model
    grouping_model = SentenceTransformer('saved_models/grouping/binary')

    # Load classification model
    classification_tokenizer = BertTokenizerFast.from_pretrained('saved_models/classification/')
    classification_model = BertForSequenceClassification.from_pretrained('saved_models/classification/')

    # Load summarization model
    summarization_tokenizer = T5Tokenizer.from_pretrained('saved_models/summarization/')
    summarization_model = T5ForConditionalGeneration.from_pretrained('saved_models/summarization/')

    return grouping_model, classification_tokenizer, classification_model, summarization_tokenizer, summarization_model

def load_topics():
    topics = pd.read_csv('data/topics.csv')
    return topics
