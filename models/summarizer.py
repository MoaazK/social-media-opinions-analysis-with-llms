from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Summarizer:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_conclusion(self, topic_text, opinions_text):
        input_text = f"Topic: {topic_text}\nOpinions: {opinions_text}\nConclusion:"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
        conclusion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return conclusion
