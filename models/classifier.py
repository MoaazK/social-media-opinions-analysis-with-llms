import torch


def classify_opinions(classification_model, classification_tokenizer, opinions, topics):
    # Map topic IDs to topic texts
    topic_id_to_text = dict(zip(topics['topic_id'], topics['text']))

    inputs = []
    for idx, row in opinions.iterrows():
        topic_text = topic_id_to_text.get(row['assigned_topic_id'], '')
        opinion_text = row['text']
        inputs.append((topic_text, opinion_text))

    tokenized_inputs = classification_tokenizer.batch_encode_plus(
        inputs,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classification_model.to(device)
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

    with torch.no_grad():
        outputs = classification_model(**tokenized_inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    id_to_label = {0: 'Opposing', 1: 'Supportive'}
    predicted_labels = [id_to_label[pred.item()] for pred in predictions]

    opinions['classification'] = predicted_labels

    return opinions
