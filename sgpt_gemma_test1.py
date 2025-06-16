from transformers import pipeline
from llama_cpp import Llama
import os

# === 1. Load Sentiment Analysis Pipeline ===
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# === 2. Load Local GGUF Model ===
model_path = "D:/AI/lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf"  # 🔁 Update this to your actual path

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=20,  # Or adjust to 0 if running on CPU
    verbose=False
)

# === 3. Star Rating → Emotion/Tone Map ===
emotion_map = {
    '1 star': ('Depression, Despair, Anger, Hopelessness, Hurt', 'Crisis/SOS'),
    '2 stars': ('Frustration, Disappointment, Loneliness, Anxiety', 'Support Needed'),
    '3 stars': ('Confusion, Acceptance, Calm, Resignation, Curiosity', 'Balanced/Listening'),
    '4 stars': ('Hope, Satisfaction, Relief, Gratitude, Encouragement', 'Affirming/Supportive'),
    '5 stars': ('Joy, Love, Pride, Inspiration, Excitement', 'Uplifting/Celebratory'),
}

def get_emotion_label(star_label):
    label = star_label.lower()
    return emotion_map.get(label, ('Calm', 'Balanced/Listening'))

# === 4. Generate Empathetic Reply using GGUF ===
def generate_response(user_text, emotion, tone):
    prompt = f"""<start_of_turn>user
The user is feeling the following emotions: {emotion}
Tone category: {tone}
They said: "{user_text}"
Respond empathetically to the user's emotional state, offering support, comfort or encouragement as needed.
<end_of_turn>
<start_of_turn>model
"""
    output = llm(prompt, max_tokens=200, temperature=0.7, stop=["<end_of_turn>"])
    reply = output["choices"][0]["text"].strip()
    return reply

# === 5. Chat Loop ===
print("Emotional Health Chatbot (type 'exit' to quit)")
print("--------------------------------------------------")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Take care! I'm here if you need me. 💙")
        break

    sentiment = sentiment_pipeline(user_input)[0]
    stars = sentiment["label"]
    emotion, tone = get_emotion_label(stars)

    print(f"Sentiment Rating: {stars}")
    print(f"Emotion(s): {emotion}")
    print(f"Tone Category: {tone}")

    bot_response = generate_response(user_input, emotion, tone)
    print(f"\nChatbot: {bot_response}\n")
