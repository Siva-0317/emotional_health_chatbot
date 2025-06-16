from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

#text = "I finally got the job! I’m so excited and proud of myself."
#text="I don’t feel like getting out of bed anymore..."
text="I have an exam tomorrow and my heart won't stop racing."

result = sentiment_pipeline(text)

print(result)
