# app.py

import logging
from flask import Flask, request, render_template, jsonify
from chatbot import create_chatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize chatbot with error handling
try:
    chatbot = create_chatbot()
    logger.info("Chatbot initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    chatbot = None

@app.route("/", methods=["GET", "POST"])
def chat():
    if not chatbot:
        return jsonify({"status": "error", "message": "Chatbot initialization failed. Check server logs."}), 500

    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if not user_input:
            return jsonify({"status": "error", "message": "Message cannot be empty."}), 400

        try:
            response = chatbot.get_response(user_input)
            logger.info(f"User input: {user_input} | Bot response: {response}")
            return jsonify({"status": "success", "response": response})
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({"status": "error", "message": "An error occurred. Please try again."}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)