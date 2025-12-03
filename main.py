# chatbot_with_sentiment_fixed.py
import os
import sys
import random
from tkinter import *
from tkinter import messagebox

# --- NLTK / VADER setup (robust) ---
import nltk
try:
    # check if resource exists, otherwise download
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# --- ChatterBot setup (safe) ---
try:
    from chatterbot import ChatBot
    from chatterbot.trainers import ListTrainer
except Exception as e:
    print("Warning: chatterbot import failed. Make sure 'chatterbot' is installed and compatible with your Python version.")
    print("Error:", e)
    # Fail gracefully: create a minimal fallback simple-bot replacement
    ChatBot = None
    ListTrainer = None

bot = None
if ChatBot is not None:
    try:
        bot = ChatBot(
            'MyChatBot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///mychatbot_database.sqlite3',
            logic_adapters=['chatterbot.logic.BestMatch'],
            read_only=False
        )
    except TypeError as e:
        # fallback to simpler instantiation if kwargs are incompatible
        try:
            print("ChatBot init with detailed args failed, trying simple fallback...", e)
            bot = ChatBot('MyChatBot')
        except Exception as e2:
            print("ChatBot fallback init failed:", e2)
            bot = None
    except Exception as e:
        print("ChatBot init error:", e)
        bot = None

# --- If ChatterBot isn't available, we provide a tiny rule-based fallback bot ---
def fallback_get_response(text_input):
    # very small rule-based mapping using your conversation list
    mapping = {
        "hello": "Hello! Welcome to our textile and clothing store. How may I assist you today?",
        "hi": "Hi there! How can I help you with fabrics or clothing today?",
        "what are the latest clothing trends?": "Current trends include breathable fabrics like cotton and linen, oversized fits, pastel colors, and sustainable eco-friendly materials.",
        "which fabric is best for summer?": "For summer, lightweight cotton, linen, and rayon fabrics are excellent due to their breathability and soft texture.",
        "thank you": "You're welcome! If you need any more help, feel free to ask.",
        "thanks": "My pleasure! Let me know if you have any other questions.",
        "goodbye": "Goodbye! Have a great day and feel free to visit again."
    }
    key = text_input.strip().lower()
    return mapping.get(key, "Sorry, I don't have an answer for that yet. Could you try rephrasing?")

# --- Example conversation list (kept small here; use your full list if trainer exists) ---
conversation = [
    "Hello", "Hello! Welcome to our textile and clothing store. How may I assist you today?",
    "Hi", "Hi there! How can I help you with fabrics or clothing today?",
    "Thank you", "You're welcome! If you need any more help, feel free to ask.",
    "Goodbye", "Goodbye! Have a great day and feel free to visit again."
]

TRAIN_FLAG = "bot_trained.flag"

def train_bot():
    if bot is None or ListTrainer is None:
        print("Skipping training: ChatterBot not available; using fallback responses.")
        return
    try:
        trainer = ListTrainer(bot)
        trainer.train(conversation)
        with open(TRAIN_FLAG, "w") as f:
            f.write("trained")
        print("Training completed.")
    except Exception as e:
        print("Training error:", e)

if not os.path.exists(TRAIN_FLAG):
    train_bot()
else:
    print("Bot already trained. Skipping training.")

# --- Tkinter UI ---
main = Tk()
main.geometry("580x740")
main.title("Textile ChatBot (with Sentiment)")

# Optional image load (safe) - uses script directory for absolute path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "images1.png")
    if os.path.isfile(img_path):
        img = PhotoImage(file=img_path)
        photoL = Label(main, image=img)
        photoL.pack(pady=5)
    else:
        print("images1.png not found at:", img_path)
except Exception as e:
    print("Image load failed:", e)

frame = Frame(main)
sc = Scrollbar(frame)
msgs = Listbox(frame, width=80, height=22, yscrollcommand=sc.set)
sc.config(command=msgs.yview)
sc.pack(side=RIGHT, fill=Y)
msgs.pack(side=LEFT, fill=BOTH, pady=10)
frame.pack(padx=10, pady=5)

text = Entry(main, font=("Verdana", 16))
text.pack(fill=X, padx=10, pady=10)
text.focus_set()

def insert_message(sender, message, sentiment=None):
    if sentiment is None:
        msgs.insert(END, f"{sender}: {message}")
    else:
        label, score = sentiment
        msgs.insert(END, f"{sender}: {message}   [{label} {score:.2f}]")
    msgs.insert(END, "------------------------------")
    msgs.yview(END)

def get_sentiment_label(text_input):
    scores = sia.polarity_scores(text_input)
    compound = scores['compound']
    if compound >= 0.5:
        return ("Positive", compound)
    elif compound <= -0.5:
        return ("Negative", compound)
    else:
        return ("Neutral", compound)

NEGATIVE_PROMPTS = [
    "I'm sorry to hear that — can I help make it right?",
    "That sounds frustrating. Tell me more so I can assist.",
    "I understand — would you like to speak to support or get a return?"
]

POSITIVE_PROMPTS = [
    "That's great! Glad you liked it 😊",
    "Awesome! Would you like recommendations based on that?",
    "Happy to hear that — I can suggest similar items."
]

def choose_sentiment_reply(label):
    if label == "Negative":
        return random.choice(NEGATIVE_PROMPTS)
    elif label == "Positive":
        return random.choice(POSITIVE_PROMPTS)
    else:
        return None

def ask_from_bot():
    user_input = text.get().strip()
    if not user_input:
        return
    sentiment = get_sentiment_label(user_input)
    insert_message("You", user_input, sentiment)
    text.delete(0, END)

    pref = choose_sentiment_reply(sentiment[0])

    try:
        if bot is not None:
            response = bot.get_response(user_input)
            bot_reply = str(response)
        else:
            bot_reply = fallback_get_response(user_input)

        if pref:
            final_reply = f"{pref}\n{bot_reply}"
        else:
            final_reply = bot_reply

        insert_message("Bot", final_reply, sentiment=(sentiment[0], sentiment[1]))
    except Exception as e:
        print("Error getting response:", e)
        # fallback to rule-based reply
        insert_message("Bot", fallback_get_response(user_input), sentiment=None)

btn = Button(main, text="Ask from Bot", font=("Verdana", 14), command=ask_from_bot)
btn.pack(pady=5)

def enter_function(event):
    btn.invoke()

main.bind('<Return>', enter_function)

def on_closing():
    main.destroy()

main.protocol("WM_DELETE_WINDOW", on_closing)
main.mainloop()
