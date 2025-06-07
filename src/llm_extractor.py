import os
import re
import warnings
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
warnings.filterwarnings("ignore")
MODEL_NAME = "yiyanghkust/finbert-tone"

# Load tokenizer and model
print("Loading FinBERT model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)


def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    chunks = [tokens[i:i + max_tokens]
              for i in range(0, len(tokens), max_tokens)]
    safe_chunks = [chunk[:max_tokens] for chunk in chunks]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in safe_chunks]


def aggregate_sentiment(chunk_preds):
    if not chunk_preds:
        return ("Neutral", 0.0)

    label_scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    for pred in chunk_preds:
        label = pred["label"]
        score = pred["score"]
        label_scores[label] += score

    total = sum(label_scores.values())
    for label in label_scores:
        label_scores[label] /= total

    final_label = max(label_scores, key=label_scores.get)
    final_score = label_scores[final_label]

    return final_label, final_score


def extract_sentiment(texts):
    results = []
    for text in texts:
        try:
            chunks = chunk_text(text)
            chunk_preds = sentiment_pipeline(chunks)
            label, score = aggregate_sentiment(chunk_preds)
            results.append({"label": label, "score": score})
        except Exception as e:
            print(f"Error processing text: {e}")
            results.append({"label": "ERROR", "score": 0.0})
    return results


def extract_leadership_sections(text):
    speaker_blocks = re.split(
        r"\n(?=[A-Z][^\n]+ — (CEO|CFO|Chief|President):)", text)
    tagged_blocks = []
    for block in speaker_blocks:
        if any(title in block for title in ["CEO", "CFO", "Chief", "President"]):
            tagged_blocks.append(block)
    return "\n".join(tagged_blocks)


if __name__ == "__main__":
    from ingest import load_transcripts
    df = load_transcripts()
    df_sample = df.copy()  # small test batch

    # sentiments = extract_sentiment(df_sample["text"].tolist())

    df_sample["ceo_cfo_text"] = df_sample["text"].apply(
        extract_leadership_sections)
    sentiments = extract_sentiment(df_sample["ceo_cfo_text"].tolist())
    df_sample["sentiment"] = [s["label"] for s in sentiments]
    df_sample["score"] = [s["score"] for s in sentiments]

    os.makedirs("data/processed/", exist_ok=True)
    df_sample[["ticker", "date", "sentiment", "score"]].to_csv(
        "data/processed/sample_sentiments.csv", index=False)

    print("\nSaved to data/processed/ceo_sentiments.csv")
    print(df_sample[["ticker", "date", "sentiment", "score"]])
