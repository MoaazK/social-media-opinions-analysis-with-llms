import pandas as pd
from models.classifier import classify_opinions
from models.summarizer import generate_summaries
from models.topic_matcher import match_topics
from utils.data_preprocessing import clean_text
from utils.model_utils import load_models, load_topics


def main():
    grouping_model, classification_tokenizer, classification_model, summarization_tokenizer, summarization_model = load_models()

    topics = load_topics()

    new_opinions = pd.DataFrame({
        "text": [
            "I believe that the face on Mars is a natural formation.",
            "I think the FACS has a good chance of changing the future in a possitive way.",
            "They could be happy, sad, surprise, angery, disgusted,and afraid.",
            "Mars has many mysteries that we have yet to uncover.",
            "I think space exploration is a waste of resources.",
            "Joining an extracurricular activity is a great way to make new friends, because you get to meet people that you would of never talked to if you didn't join that club."
        ]
    })

    topics['text'] = topics['text'].apply(clean_text)
    new_opinions['text'] = new_opinions['text'].apply(clean_text)

    # Step 1: Match/Group topics
    new_opinions = match_topics(grouping_model, topics, new_opinions)

    # Step 2: Classify opinions
    new_opinions = classify_opinions(classification_model, classification_tokenizer, new_opinions, topics)

    # Step 3: Generate summaries
    topic_summaries = generate_summaries(summarization_model, summarization_tokenizer, new_opinions, topics)

    # Output the summaries
    for topic_id, summary in topic_summaries.items():
        topic_text = topics.loc[topics['topic_id'] == topic_id, 'text'].values[0]
        print(f"Topic ID: {topic_id}")
        print(f"Topic: {topic_text}")
        print(f"Summary: {summary}")
        print('-' * 80)

if __name__ == '__main__':
    main()
