import os
import openai
import telebot
from dotenv import load_dotenv

stress_count = 0
conversation_history = ""

# A simple in-memory data structure to store stress keyword counts
# In a production environment, consider using a more persistent storage, like a database
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

# Load API keys from the .env file
load_dotenv("keys.env")
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI API and the Telegram bot
openai.api_key = OPENAI_API_KEY
bot = telebot.TeleBot(TELEGRAM_API_KEY)

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hello, how are you doing?")
    global stress_count
    stress_count = 0
    global conversation_history 
    conversation_history = ""

def analyze_stress_level(user_message):
    global stress_count
    stress_keyword_count = sum([keyword in user_message.lower() for keyword in STRESS_KEYWORDS])
    stress_count = stress_count + stress_keyword_count
    
def generate_gpt_response(user_message):
    global conversation_history
    analyze_stress_level(user_message)
    conversation_history += f"User: {user_message}\nAI:"
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=f"You are a friendly and casual AI who talks like a human friend.\n{conversation_history}",
    #     temperature=0.8,
    #     max_tokens=50,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )
    # response_text = response.choices[0].text.strip()
    response_text = 'I heard you'
    conversation_history += f" {response_text}\n"
    print(conversation_history)

    # Check if the stress keyword count has reached the threshold
    if stress_count >= STRESS_KEYWORD_THRESHOLD:
        response_text += HIGH_STRESS_MESSAGE
    return response_text

@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text_message(message):
    user_text = message.text
    bot_response = generate_gpt_response(user_text)
    bot.send_message(message.chat.id, bot_response)

# Start the bot's polling loop
bot.polling()