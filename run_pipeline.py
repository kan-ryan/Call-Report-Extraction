from transformers import BertTokenizer, BertForSequenceClassification, pipeline

MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

text = """Thanks, Nancy, and good afternoon, everyone... Music revenue has now hit an inflection point after many quarters of decline."""
result = sentiment_pipeline(text)
print(result)