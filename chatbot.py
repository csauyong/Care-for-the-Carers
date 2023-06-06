import os
import openai
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv

# A simple in-memory data structure to store stress keyword counts and chat history
# In a production environment, consider using a more persistent storage, like a database
stress_counts = {}
conversation_history = {}

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
HIGH_STRESS_MESSAGE = "I understand your situation. Being a caregiver could be exhausting and may affect you mentally. Don't forget your well-being before taking care of others. In addition to seeking support from family and friends, there are also support groups in Hong Kong for caregivers of people with dementia. Would you like me to refer you to some of these groups?"

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
    stress_counts[message.chat.id] = 0
    conversation_history[message.chat.id] = ""

def analyze_stress_level(user_message, user_id, stress_counts):
    stress_keyword_count = sum([keyword in user_message.lower() for keyword in STRESS_KEYWORDS])
    stress_counts[user_id] += stress_keyword_count
    
def generate_gpt_response(user_message, user_id, stress_counts, conversation_history):
    analyze_stress_level(user_message, user_id, stress_counts)
    conversation_history[user_id] += f"User: {user_message}\nAI:"
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
    response_text = f"You are user {user_id}. You have mentioned {stress_counts[user_id]} keywords."
    conversation_history[user_id] += f" {response_text}\n"
    print(conversation_history)

    # Check if the stress keyword count has reached the threshold
    stressed = False
    if stress_counts[user_id] >= STRESS_KEYWORD_THRESHOLD:
        response_text += HIGH_STRESS_MESSAGE
        stressed = True
    return response_text, stressed

@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text_message(message):
    user_text = message.text
    chat_id = message.chat.id
    bot_response, stressed = generate_gpt_response(user_text, chat_id, stress_counts, conversation_history)
    if stressed:
        markup = InlineKeyboardMarkup()
        markup.row_width = 2
        markup.add(InlineKeyboardButton("Yes", callback_data="yes"), InlineKeyboardButton("No", callback_data="no"))
        bot.send_message(chat_id, bot_response, reply_markup=markup)
    else:
        bot.send_message(chat_id, bot_response)

@bot.callback_query_handler(func=lambda call: True)
def query_handler(call):
    bot.answer_callback_query(call.id)
    answer = call.data
    chat_id = call.message.chat.id

    if answer == 'yes':
        bot.send_message(chat_id, "Great. Here are a few support groups in Hong Kong that you may find helpful:\nThe Hong Kong Alzheimer's Disease Association: http://www.hkalzheimers.org/\nThe Carers Support Group: https://www.carers.org.hk/\nThe Mental Health Association of Hong Kong: https://www.mhahk.org.hk/\nI hope these resources can provide you with the support and guidance you need during this challenging time. Please don't hesitate to reach out if you need further assistance.")
    else:
        bot.send_message(chat_id, 'It alright. I am always here for you.')

# Start the bot's polling loop
bot.polling()