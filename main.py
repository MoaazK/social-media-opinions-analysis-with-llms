from utils.data_preprocessing import preprocess_data, split_data
from models.topic_matcher import TopicMatcher
from models.comment_classifier import OpinionClassifier
from models.summarizer import Summarizer

def main():
    # Load and preprocess data
    topics, opinions, conclusions = preprocess_data()
    train_data, test_data = split_data(opinions)

    # Match topics and opinions
    topic_matcher = TopicMatcher()
    topic_matcher.fine_tune(topics, train_data)
    matched_opinions = topic_matcher.match_topics(topics, test_data)
    topic_matcher.evaluate(test_data, matched_opinions)

    # Classify opinions
    classifier = OpinionClassifier()
    # Assuming the classifier is trained elsewhere and loaded here
    predictions = classifier.predict(matched_opinions['clean_text'].tolist())
    matched_opinions['classification'] = predictions

    # Generate conclusions
    summarizer = Summarizer()
    for topic_id in matched_opinions['matched_topic'].unique():
        topic_text = topics.iloc[topic_id]['text']
        opinions_text = ' '.join(matched_opinions[matched_opinions['matched_topic'] == topic_id]['text'].tolist())
        conclusion = summarizer.generate_conclusion(topic_text, opinions_text)
        print(f"Conclusion for Topic {topic_id}:\n{conclusion}\n")

if __name__ == '__main__':
    main()
