import torch


def generate_summaries(summarization_model, summarization_tokenizer, opinions, topics):
    topic_summaries = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summarization_model.to(device)

    for topic_id in topics['topic_id']:
        topic_text = topics.loc[topics['topic_id'] == topic_id, 'text'].values[0]
        topic_opinions = opinions[opinions['assigned_topic_id'] == topic_id]

        if topic_opinions.empty:
            continue

        opinions_text = '\n'.join(topic_opinions['text'].tolist())

        # Prepare input text for summarization
        input_text = 'summarize: ' + topic_text + '\n' + opinions_text

        input_ids = summarization_tokenizer.encode(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(device)

        summary_ids = summarization_model.generate(
            input_ids=input_ids,
            max_length=150,
            num_beams=5,
            early_stopping=True
        )
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        topic_summaries[topic_id] = summary

    return topic_summaries
