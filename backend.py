from flask import Flask, request, jsonify
from flask_cors import CORS

# import your chatbot code here
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

app = Flask(__name__)
CORS(app)

# initialize chatbot
bot = ChatBot("MyChatbot")
trainer = ListTrainer(bot)
trainer.train(["Hello", "Hi there!", "How are you?", "I'm fine!"])

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    bot_reply = str(bot.get_response(user_msg))
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
