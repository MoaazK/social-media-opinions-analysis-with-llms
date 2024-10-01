from sentence_transformers import SentenceTransformer, InputExample, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from datasets import Dataset


class TopicMatcher:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def prepare_data(self, topics: pd.DataFrame, opinions: pd.DataFrame, n=1):
        # data = {
        #     'anchor': [],
        #     'positive': [],
        #     'negative': [],
        #     # 'label': []
        # }

        data = {
            'text1': [],
            'text2': [],
            'label': []
        }

        for _, opinion in opinions.iterrows():
            matching_topics = topics[topics['topic_id'] == opinion['topic_id']]['text']
            if not matching_topics.empty:
                topic_text = matching_topics.values[0]
                # data['anchor'].append(opinion['text'])
                # data['positive'].append(topic_text)
                data['text1'].append(opinion['text'])
                data['text2'].append(topic_text)
                data['label'].append(1)

                # Negative samples (pair with other topics)
                negative_topics = topics[topics['topic_id'] != opinion['topic_id']]['text'].sample(n=n).values
                for neg_topic in negative_topics:
                    # data['negative'].append(neg_topic)
                    data['text1'].append(opinion['text'])
                    data['text2'].append(neg_topic)
                    data['label'].append(0)

        data = Dataset.from_dict(data)

        return data

        # # Define the loss function
        # train_loss = losses.CosineSimilarityLoss(self.model)

        # # Define training arguments
        # training_args = SentenceTransformerTrainingArguments(
        #     output_dir="./sentence_transformer_output",
        #     num_train_epochs=1,
        #     per_device_train_batch_size=16,
        #     warmup_steps=100,
        #     eval_strategy="no",
        #     save_strategy="no",
        #     learning_rate=2e-5,
        # )

        # # Initialize the trainer
        # trainer = SentenceTransformerTrainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     train_loss=train_loss,
        # )

        # # Train the model
        # trainer.train()

        # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        # train_loss = losses.CosineSimilarityLoss(self.model)

        # self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    def match_topics(self, topics: pd.DataFrame, opinions: pd.DataFrame):
        topic_embeddings = self.model.encode(topics['text'].tolist())
        opinion_embeddings = self.model.encode(opinions['text'].tolist())

        similarity_matrix = cosine_similarity(opinion_embeddings, topic_embeddings)

        matches = np.argmax(similarity_matrix, axis=1)
        opinions['matched_topic'] = matches
        opinions['similarity_score'] = np.max(similarity_matrix, axis=1)
        return opinions

    # def evaluate(self, true_opinions, predicted_opinions):
    #     true_labels = true_opinions.index.tolist()
    #     pred_labels = predicted_opinions['matched_topic']

    #     accuracy = accuracy_score(true_labels, pred_labels)
    #     precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    #     return {
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1': f1
    #     }
