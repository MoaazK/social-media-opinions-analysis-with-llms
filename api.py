from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
from models.classifier import classify_opinions
from models.summarizer import generate_summaries
from models.topic_matcher import match_topics
from utils.data_preprocessing import clean_text
from utils.model_utils import load_models, load_topics

app = FastAPI()

grouping_model, classification_tokenizer, classification_model, summarization_tokenizer, summarization_model = load_models()
topics = load_topics()
topics['text'] = topics['text'].apply(clean_text)

class OpinionRequest(BaseModel):
    texts: list

@app.post('/process_opinions')
def process_opinions(request: OpinionRequest):
    new_opinions = pd.DataFrame({'text': request.texts})
    new_opinions['text'] = new_opinions['text'].apply(clean_text)

    new_opinions = match_topics(grouping_model, topics, new_opinions)
    new_opinions = classify_opinions(classification_model, classification_tokenizer, new_opinions, topics)
    topic_summaries = generate_summaries(summarization_model, summarization_tokenizer, new_opinions, topics)

    response = []
    for topic_id, summary in topic_summaries.items():
        topic_text = topics.loc[topics['topic_id'] == topic_id, 'text'].values[0]
        response.append({
            'topic_id': topic_id,
            'topic': topic_text,
            'summary': summary
        })

    return {'results': response}
