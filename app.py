from flask import Flask, render_template, request, jsonify
from bot import HybridAssistant, wishMe
import markdown  # To convert markdown to HTML

# API Keys
GOOGLE_API_KEY = 'AIzaSyAJ5xVpjeaCXWHTjqg_eQ3TAuZ82QPmJWM'
WEATHER_API_KEY = '7bcc14a012464257fb28ff3ee878bd39'
NEWS_API_KEY = '1f83a0479f694f0091098c6bc25bcfe8'

# Bot setup
assistant = HybridAssistant(GOOGLE_API_KEY, WEATHER_API_KEY, NEWS_API_KEY)
chat_history = []

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    greeting = wishMe()
    return render_template('pr.html', greeting=greeting)

# Chat route
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('message')
    if not user_input:
        return jsonify({'response': 'Please enter a message.'})

    display_response, _ = assistant.process_input(user_input)

    # Convert Markdown to HTML
    display_response_html = markdown.markdown(display_response)

    chat_history.append({"user": user_input, "bot": display_response_html})
    return jsonify({'response': display_response_html})

if __name__ == "__main__":
    app.run(debug=True)
    #test
