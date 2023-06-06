import os
import openai
from flask import Flask, request, jsonify
from telegram import Bot, Update

# Replace with your Telegram bot token obtained from BotFather
TELEGRAM_BOT_TOKEN = '6059193114:AAEhxBZGBao1NRANcRzDhDL7aMlGJ0-JEw0'
# Replace with your OpenAI API key
OPENAI_API_KEY = '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'

# A simple in-memory data structure to store stress keyword counts
# In a production environment, consider using a more persistent storage, like a database
USER_STRESS_KEYWORD_COUNTS = {}
USER_SESSIONS = {}
STRESS_KEYWORDS = [
    'overwhelmed', 'anxious', 'stressed', 'tense', 'exhausted', 'worried',
    'nervous', 'restless', 'irritable', 'burned out', 'burnout', 'fatigued',
    'frustrated', 'upset', 'distressed', 'hopeless', 'helpless', 'desperate',
    'trapped', 'panicked', 'pressure', 'strained', 'agitated', 'on edge',
    'difficulty concentrating', "can't focus", "can't sleep", "losing sleep",
    'sleepless', 'insomnia', 'racing thoughts', 'anxiety', 'depressed', 'depression',
    'crying', 'mood swings', "can't relax", "feeling low", "lack of motivation",
    'unmotivated', 'procrastinating', 'overthinking', 'doubtful', 'uncertain'
]
STRESS_KEYWORD_THRESHOLD = 3
HIGH_STRESS_MESSAGE = "\n\nIt seems like you're experiencing a high level of stress. I recommend talking to a mental health professional or a counselor for further guidance and support."

app = Flask(__name__)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Initialize the OpenAI API client
openai.api_key = OPENAI_API_KEY

def analyze_stress_level(user_id, user_message):
    stress_keyword_count = sum([keyword in user_message.lower() for keyword in STRESS_KEYWORDS])
    USER_STRESS_KEYWORD_COUNTS[user_id] = USER_STRESS_KEYWORD_COUNTS.get(user_id, 0) + stress_keyword_count

def generate_response(user_id, user_message):
    analyze_stress_level(user_id, user_message)

    conversation_history = USER_SESSIONS.get(user_id, "")
    conversation_history += f"User: {user_message}\nAI:"

    # Generate a response using ChatGPT
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"You are a friendly and casual AI who talks like a human friend.\n{conversation_history}",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.8,
    )

    response_text = response.choices[0].text.strip()

    conversation_history += f" {response_text}\n"
    USER_SESSIONS[user_id] = conversation_history

    # Check if the stress keyword count has reached the threshold
    if USER_STRESS_KEYWORD_COUNTS[user_id] >= STRESS_KEYWORD_THRESHOLD:
        response_text += HIGH_STRESS_MESSAGE

    return response_text

def process_message(update: Update):
    chat_id = update.effective_chat.id
    user_message = update.effective_message.text

    # Generate a response using ChatGPT with conversation context
    response_message = generate_response(chat_id, user_message)

    bot.send_message(chat_id=chat_id, text="hi", parse_mode="HTML")
    #bot.send_message(chat_id=chat_id, text=response_message, parse_mode="HTML")

@app.route('/webhook', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    process_message(update)
    return jsonify(status='ok')

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)), host='0.0.0.0')