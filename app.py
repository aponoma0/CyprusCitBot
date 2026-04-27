"""This file runs a small Telegram study bot with an About button.
Edit it when you change bot messages, commands, or button behavior.
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
- Use English or Russian depending on user language
- Focus on Cyprus history, geography, government, culture
- Be short and useful
"""

ABOUT_BUTTON = "About"


def build_main_keyboard() -> ReplyKeyboardMarkup:
    """Create the simple keyboard shown under the chat."""
    return ReplyKeyboardMarkup([[ABOUT_BUTTON]], resize_keyboard=True)


def get_start_text() -> str:
    """Return the welcome text shown after /start."""
    return (
        "Cyprus Citizenship Helper Bot\n\n"
        "I can help you prepare for the naturalization exam.\n"
        "Ask me about history, geography, or the political system.\n\n"
        "Tap About to learn what this bot does."
    )


def get_about_text() -> str:
    """Return the About message for the About button."""
    return (
        "About this bot\n\n"
        "This bot helps you study for the Cyprus citizenship exam.\n"
        "It can explain Cyprus history, geography, government, and culture in simple words.\n"
        "You can write in English or Russian."
    )


# ======================
# AI LOGIC
# ======================
def query_ai(user_text: str) -> str:
    """Send the user question to the AI model and return the answer."""
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as error:
        logger.error("AI error: %s", error)
        return "AI is temporarily unavailable. Please try again in a moment."


# ======================
# TELEGRAM HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the welcome message and show the main keyboard."""
    if update.message is None:
        return

    await update.message.reply_text(get_start_text(), reply_markup=build_main_keyboard())


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the About text and keep the keyboard visible."""
    if update.message is None:
        return

    await update.message.reply_text(get_about_text(), reply_markup=build_main_keyboard())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle normal messages and the About button."""
    if update.message is None:
        return

    user_text = update.message.text
    if not user_text:
        return

    if user_text.strip().lower() == ABOUT_BUTTON.lower():
        await about(update, context)
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = query_ai(user_text)
    await update.message.reply_text(response, reply_markup=build_main_keyboard())


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
