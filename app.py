"""
app.py - Flask backend + chatbot (ChatterBot if available, fallback otherwise)
Run: python app.py
Endpoint:
  POST /api/chat  { "message": "Hello" }  -> { "reply": "...", "sentiment": {"label":"Positive","score":0.7} }

Notes:
- If chatterbot is not installed or not compatible with your Python version,
  the server will run using a rule-based fallback so the frontend still works.
- Make sure to `pip install -r requirements.txt` before running.
"""
import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- NLTK / VADER setup (ensure resource is present) ---
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logging.info("vader_lexicon not found — downloading...")
    nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# --- Try importing ChatterBot, but fall back gracefully if import fails ---
use_chatterbot = False
bot = None
trainer = None

try:
    from chatterbot import ChatBot
    from chatterbot.trainers import ListTrainer
    use_chatterbot = True
    logging.info("ChatterBot import succeeded.")
except Exception as e:
    logging.warning("ChatterBot import failed or incompatible with Python version. Falling back to rule-based bot. Error: %s", e)
    use_chatterbot = False

# --- Conversation list (useful for training or fallback replies) ---
CONVERSATION = [
    "Hello", "Hello! Welcome to our textile and clothing store. How may I assist you today?",
    "Hi", "Hi there! How can I help you with fabrics or clothing today?",
    "Good morning", "Good morning! How can I support you with your textile needs today?",
    "Good afternoon", "Good afternoon! How may I assist you?",
    "What are the latest clothing trends?", "Current trends include breathable fabrics like cotton and linen, oversized fits, pastel colors, and sustainable eco-friendly materials.",
    "What are the latest textile trends?", "Popular textile trends include organic fabrics, textured weaves, geometric patterns, digital prints, and soft-tone dyeing techniques.",
    "What about textile patterns?", "Floral prints, geometric motifs, minimalistic stripes, and classy jacquard textures are trending this season.",
    "What fabrics do you recommend?", "I recommend organic cotton for comfort, linen blends for summer wear, and OEKO-TEX certified materials for safe and sustainable clothing.",
    "Which fabric is best for summer?", "For summer, lightweight cotton, linen, and rayon fabrics are excellent due to their breathability and soft texture.",
    "Which fabric is best for winter?", "For winter, wool, fleece, heavy cotton, and flannel are highly suitable for warmth and comfort.",
    "Do you offer custom tailoring?", "Yes, we offer tailoring for selected items. Please share your measurements or preferred fit for assistance.",
    "Do you provide size guidance?", "Yes, we do! We offer sizes from XS to XXL, and I can help you choose the best fit based on your body measurements.",
    "What sizes do you offer?", "We offer sizes XS, S, M, L, XL, and XXL. For detailed measurements, please check our size chart.",
    "Can I return an item?", "Yes, items can be returned within 30 days, provided they are unused and have original tags attached.",
    "How do I return an item?", "To return an item, simply submit a return request through your order history or contact our support team. We will guide you through the steps.",
    "Do you accept exchanges?", "Yes, we accept exchanges within 30 days. The item should be unused and in its original condition.",
    "What payment methods do you accept?", "We accept credit cards, debit cards, UPI, PayPal, and net banking. For bulk orders, bank transfers are also supported.",
    "Do you offer discounts?", "Yes, we frequently offer seasonal sales, promotional codes, and special offers for loyalty members.",
    "Are there any ongoing offers?", "Currently, we have discounts on selected cotton fabrics and a 10% promotion on all linen products.",
    "Do you offer wholesale or bulk pricing?", "Yes, we do! For bulk or wholesale orders, please provide your quantity and we will share a customized quotation.",
    "Do you offer international shipping?", "Yes, we ship internationally. Charges and delivery times vary by location.",
    "How long does delivery take?", "Standard delivery takes 4–7 days, while express delivery takes 1–3 days depending on your region.",
    "What is your shipping policy?", "We offer free shipping on selected orders and charge a small fee for express delivery. Shipping details are shown at checkout.",
    "Is COD available?", "Yes, Cash on Delivery is available for select regions and order values.",
    "Are your fabrics sustainable?", "Yes, we offer a wide range of sustainable fabrics, including organic cotton, bamboo fabrics, and recycled polyester blends.",
    "How should I wash cotton fabric?", "For cotton fabrics, use mild detergent and cold water. Avoid direct sunlight for drying if you want to preserve color.",
    "How should I wash linen?", "Wash linen with gentle detergent in lukewarm water. Avoid harsh rubbing to maintain fabric texture.",
    "Do your fabrics shrink?", "Some natural fabrics like cotton and linen may have slight shrinkage. We recommend pre-washing before tailoring.",
    "Can I track my order?", "Yes, after placing your order, you will receive a tracking link via email or SMS.",
    "Do you provide sample swatches?", "Yes, we offer fabric swatches for selected materials so you can check texture and color before purchasing.",
    "Can you recommend a fabric for formal wear?", "For formal wear, we suggest premium cotton, satin blends, silk blends, and wrinkle-resistant fabrics.",
    "Can you recommend a fabric for kids?", "For kids, soft and breathable fabrics like cotton, jersey knit, and bamboo fabric are ideal.",
    "Thank you", "You're welcome! If you need any more help, feel free to ask.",
    "Thanks", "My pleasure! Let me know if you have any other questions.",
    "Goodbye", "Goodbye! Have a great day and feel free to visit again."
]

TRAIN_FLAG = "flask_bot_trained.flag"

# --- Initialize ChatterBot if available ---
if use_chatterbot:
    try:
        bot = ChatBot(
            'MyChatBot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///mychatbot_database.sqlite3',
            logic_adapters=['chatterbot.logic.BestMatch'],
            read_only=False
        )
        trainer = ListTrainer(bot)
        logging.info("ChatterBot instance created.")
    except Exception as e:
        logging.warning("Failed to initialize ChatBot with full options; attempting basic ChatBot init. Error: %s", e)
        try:
            bot = ChatBot('MyChatBot')
            trainer = ListTrainer(bot)
            logging.info("ChatterBot created with fallback init.")
        except Exception as e2:
            logging.warning("Final ChatBot init failed. Falling back to rule-based. Error: %s", e2)
            use_chatterbot = False
            bot = None
            trainer = None

# --- If ChatterBot available, perform training (once) ---
def train_chatterbot_if_needed():
    if not use_chatterbot or bot is None or trainer is None:
        return False
    try:
        if os.path.exists(TRAIN_FLAG):
            logging.info("Training flag found; skipping chatterbot training.")
            return True
        logging.info("Training ChatterBot with built-in conversation list...")
        trainer.train(CONVERSATION)
        # create a small flag so we don't retrain every server start
        with open(TRAIN_FLAG, "w") as f:
            f.write(f"trained {datetime.utcnow().isoformat()}Z")
        logging.info("Training completed and flag written.")
        return True
    except Exception as e:
        logging.error("ChatterBot training error: %s", e)
        return False

if use_chatterbot:
    train_chatterbot_if_needed()

# --- Fallback simple rule-based reply system (works without external deps) ---
import difflib
def fallback_get_response(user_text: str) -> str:
    # Basic exact-match -> reply mapping (lowercased keys)
    mapping = {}
    # Build mapping from the CONVERSATION list: pairwise entries
    for i in range(0, len(CONVERSATION)-1, 2):
        k = CONVERSATION[i].strip().lower()
        v = CONVERSATION[i+1]
        mapping[k] = v

    key = user_text.strip().lower()
    # Exact or close match
    if key in mapping:
        return mapping[key]
    # try best close match with difflib
    close = difflib.get_close_matches(key, mapping.keys(), n=1, cutoff=0.6)
    if close:
        return mapping[close[0]]
    # common fallbacks
    if any(word in key for word in ["hi", "hello", "hey"]):
        return mapping.get("hello", "Hello!")
    if any(word in key for word in ["thank", "thanks"]):
        return mapping.get("thank you", "You're welcome!")
    if "return" in key or "refund" in key:
        return mapping.get("can i return an item?", "Yes, items can be returned within 30 days, provided they are unused and have original tags attached.")
    # last resort: gentle fallback
    return "Sorry — I don't have a precise answer for that yet. Could you rephrase or ask about fabrics, sizes, orders, or returns?"

# --- Sentiment helper ---
def analyze_sentiment(text: str):
    scores = sia.polarity_scores(text)
    compound = scores.get('compound', 0.0)
    if compound >= 0.5:
        label = "Positive"
    elif compound <= -0.5:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "score": compound, "details": scores}

# --- Flask app ---
app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route("/health", methods=["GET"])
def health():
    status = {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "chatterbot_available": bool(use_chatterbot and bot is not None)
    }
    return jsonify(status)

@app.route("/train", methods=["POST"])
def train():
    """Manual endpoint to retrain chatterbot (if available)."""
    if not use_chatterbot or trainer is None:
        return jsonify({"ok": False, "error": "chatterbot not available on this server"}), 400
    try:
        trainer.train(CONVERSATION)
        # update flag timestamp
        with open(TRAIN_FLAG, "w") as f:
            f.write(f"trained_manual {datetime.utcnow().isoformat()}Z")
        return jsonify({"ok": True, "message": "trained"})
    except Exception as e:
        logging.exception("train error")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat_api():
    """
    POST body: { "message": "User message here" }
    Response: { "reply": "...", "sentiment": { "label": "...", "score": 0.12 } }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_msg = (data.get("message") or data.get("text") or "").strip()
        if not user_msg:
            return jsonify({"error": "no message provided"}), 400

        # Sentiment
        sentiment = analyze_sentiment(user_msg)

        # Get reply (ChatterBot if available, otherwise fallback)
        reply_text = None
        if use_chatterbot and bot is not None:
            try:
                response = bot.get_response(user_msg)
                reply_text = str(response)
            except Exception as e:
                logging.exception("ChatterBot get_response failed; using fallback. Error: %s", e)
                reply_text = fallback_get_response(user_msg)
        else:
            reply_text = fallback_get_response(user_msg)

        # Compose response
        out = {
            "reply": reply_text,
            "sentiment": sentiment
        }
        return jsonify(out)
    except Exception as exc:
        logging.exception("Unhandled error in /api/chat")
        return jsonify({"error": "internal server error", "details": str(exc)}), 500


if __name__ == "__main__":
    # Optionally allow setting a custom port via environment variable
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    logging.info("Starting server on %s:%s (chatterbot_available=%s)", host, port, use_chatterbot and bot is not None)
    app.run(host=host, port=port, debug=True)
