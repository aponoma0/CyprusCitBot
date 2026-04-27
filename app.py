"""This file runs a small Telegram study bot with language choice and an About button.
Edit it when you change bot messages, language flow, commands, or button behavior.
Copy it as a starting point for another simple Telegram bot."""

import asyncio
import logging
import os
import threading

from flask import Flask
from huggingface_hub import InferenceClient
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# ======================
# LOGGING CONFIG
# ======================
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# FLASK (Keep-Alive)
# ======================
server = Flask(__name__)


@server.route("/")
def health():
    return "Cyprus Citizenship Bot is Running"


def run_flask():
    """Start the small health server for Hugging Face Spaces."""
    server.run(host="0.0.0.0", port=7860)


# ======================
# ENV VARIABLES
# ======================
HF_TOKEN = os.getenv("HF_TOKEN")
TG_TOKEN = os.getenv("TG_TOKEN")

# ======================
# HUGGING FACE MODEL
# ======================
client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

SYSTEM_PROMPT = """
You are a helpful tutor for the Cyprus citizenship exam.

RULES:
- Explain simply and clearly
- Use the user's chosen language
- Focus on Cyprus history, geography, government, culture
- Be short and useful
"""

ABOUT_BUTTON = "About"
ENGLISH_BUTTON = "English"
RUSSIAN_BUTTON = "Russian"
START_MESSAGE = (
    "**Привет! Калористе! (Καλώς ορίσατε!)** 🇨🇾✨\n\n"
    "Я твой персональный тренажер для подготовки к экзамену на **гражданство Кипра**.\n\n"
    "С 2024 года правила натурализации стали строже, и знание «Современной реальности Кипра» — "
    "обязательное условие. Я помогу тебе пройти этот путь без стресса!\n\n"
    "**Что мы будем учить:**\n"
    "🔹 **История:** От древности до образования республики.\n"
    "🔹 **Политика:** Как устроена власть и кто принимает решения.\n"
    "🔹 **География:** Города, горы, реки и климат.\n"
    "🔹 **Культура:** Традиции, праздники и национальные символы.\n\n"
    "**Доступные режимы:**\n"
    "📖 *Обучение* — просматривай вопросы по категориям.\n"
    "⏱ *Экзамен* — симуляция реального теста (25 вопросов).\n\n"
    "Готов начать? Жми кнопку ниже! 👇"
)

TEXT = {
    "en": {
        "about_button": "About",
        "choose_language": "Please choose your language.",
        "welcome": (
            "Cyprus Citizenship Helper Bot\n\n"
            "I can help you prepare for the naturalization exam.\n"
            "Ask me about history, geography, or the political system.\n\n"
            "Tap About to learn what this bot does."
        ),
        "about": (
            "About this bot\n\n"
            "This bot helps you study for the Cyprus citizenship exam.\n"
            "It can explain Cyprus history, geography, government, and culture in simple words.\n"
            "You can write in English."
        ),
        "language_saved": "Language saved: English.",
        "language_needed": "Please choose a language first.",
        "ai_error": "AI is temporarily unavailable. Please try again in a moment.",
    },
    "ru": {
        "about_button": "О боте",
        "choose_language": "Пожалуйста, выбери язык.",
        "welcome": (
            "Бот-помощник для гражданства Кипра\n\n"
            "Я могу помочь тебе подготовиться к экзамену на натурализацию.\n"
            "Спроси меня об истории, географии или политической системе.\n\n"
            "Нажми О боте, чтобы узнать, что делает этот бот."
        ),
        "about": (
            "О боте\n\n"
            "Этот бот помогает готовиться к экзамену на гражданство Кипра.\n"
            "Он может простыми словами объяснить историю, географию, государственное устройство и культуру Кипра.\n"
            "Ты можешь писать по-русски."
        ),
        "language_saved": "Язык сохранен: Русский.",
        "language_needed": "Пожалуйста, сначала выбери язык.",
        "ai_error": "ИИ сейчас недоступен. Попробуй еще раз чуть позже.",
    },
}

user_languages: dict[int, str] = {}


def build_main_keyboard(language: str) -> ReplyKeyboardMarkup:
    """Create the simple keyboard shown under the chat."""
    return ReplyKeyboardMarkup([[get_text(language, "about_button")]], resize_keyboard=True)


def build_language_keyboard() -> ReplyKeyboardMarkup:
    """Create the keyboard for the first language choice."""
    return ReplyKeyboardMarkup([[ENGLISH_BUTTON, RUSSIAN_BUTTON]], resize_keyboard=True, one_time_keyboard=True)


def get_user_language(update: Update) -> str | None:
    """Return the saved language for the current user."""
    if update.effective_user is None:
        return None

    return user_languages.get(update.effective_user.id)


def get_text(language: str, key: str) -> str:
    """Return one translated message string."""
    return TEXT[language][key]


# ======================
# AI LOGIC
# ======================
def query_ai(user_text: str, language: str) -> str:
    """Send the user question to the AI model and return the answer."""
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}\nAnswer only in {'Russian' if language == 'ru' else 'English'}."},
                {"role": "user", "content": user_text},
            ],
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as error:
        logger.error("AI error: %s", error)
        return get_text(language, "ai_error")


# ======================
# TELEGRAM HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the invitation message, then ask for the user's language."""
    if update.message is None:
        return

    await update.message.reply_text(START_MESSAGE)
    await update.message.reply_text(TEXT["en"]["choose_language"], reply_markup=build_language_keyboard())


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the About text and keep the keyboard visible."""
    if update.message is None:
        return

    language = get_user_language(update)
    if language is None:
        await update.message.reply_text(TEXT["en"]["choose_language"], reply_markup=build_language_keyboard())
        return

    await update.message.reply_text(get_text(language, "about"), reply_markup=build_main_keyboard(language))


async def save_language_choice(update: Update, language: str):
    """Save the chosen language and send the welcome text."""
    if update.message is None or update.effective_user is None:
        return

    user_languages[update.effective_user.id] = language
    await update.message.reply_text(get_text(language, "language_saved"), reply_markup=build_main_keyboard(language))
    await update.message.reply_text(get_text(language, "welcome"), reply_markup=build_main_keyboard(language))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle language choice, About button, and AI messages."""
    if update.message is None:
        return

    user_text = update.message.text
    if not user_text:
        return

    clean_text = user_text.strip()

    if clean_text == ENGLISH_BUTTON:
        await save_language_choice(update, "en")
        return

    if clean_text == RUSSIAN_BUTTON:
        await save_language_choice(update, "ru")
        return

    language = get_user_language(update)
    if language is None:
        await update.message.reply_text(TEXT["en"]["choose_language"], reply_markup=build_language_keyboard())
        return

    if clean_text.lower() == get_text(language, "about_button").lower() or clean_text.lower() == ABOUT_BUTTON.lower():
        await about(update, context)
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = query_ai(user_text, language)
    await update.message.reply_text(response, reply_markup=build_main_keyboard(language))


# ======================
# MAIN EXECUTION
# ======================
async def run_bot():
    """Build and run the Telegram application."""
    if not TG_TOKEN:
        logger.error("TG_TOKEN not found in environment variables.")
        return

    app = ApplicationBuilder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    stop_event = asyncio.Event()
    await stop_event.wait()


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by system/user.")
    except Exception as error:
        logger.critical("Critical failure: %s", error)
