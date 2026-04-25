import os
import threading
import logging
import asyncio
from flask import Flask
from huggingface_hub import InferenceClient

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

# ======================
# LOGGING CONFIG
# ======================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ======================
# FLASK (Keep-Alive)
# ======================
server = Flask(__name__)


@server.route('/')
def health():
    return "Cyprus Citizenship Bot is Running 🇨🇾"


def run_flask():
    # HF Spaces use port 7860 by default
    server.run(host='0.0.0.0', port=7860)


# ======================
# ENV VARIABLES
# ======================
HF_TOKEN = os.getenv("HF_TOKEN")
TG_TOKEN = os.getenv("TG_TOKEN")

# ======================
# HUGGING FACE MODEL
# ======================
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

SYSTEM_PROMPT = """
You are a helpful tutor for the Cyprus citizenship exam.

RULES:
- Explain simply and clearly
- Use English or Russian depending on user language
- Focus on Cyprus history, geography, government, culture
- Be short and useful
"""


# ======================
# AI LOGIC
# ======================
def query_ai(user_text):
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI error: {e}")
        return "AI is temporarily unavailable. Please try again in a moment."


# ======================
# TELEGRAM HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🇨🇾 **Cyprus Citizenship Helper Bot**\n\n"
        "I can help you prepare for the naturalization exam.\n"
        "Ask me about history, geography, or the political system!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        return

    # Visual feedback: "typing..."
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    response = query_ai(user_text)
    await update.message.reply_text(response)


# ======================
# MAIN EXECUTION
# ======================
async def run_bot():
    """Build and run the Telegram application."""
    if not TG_TOKEN:
        logger.error("TG_TOKEN not found in environment variables!")
        return

    app = ApplicationBuilder().token(TG_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")

    # Using the 'run_polling' method within an async context is the simplest
    # stable way to handle the internal loop in PTB v20.x
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    # Keep the bot running until it is interrupted
    stop_event = asyncio.Event()
    await stop_event.wait()


if __name__ == "__main__":
    # 1. Start Flask in background to satisfy HF Space's port requirement
    threading.Thread(target=run_flask, daemon=True).start()

    # 2. Run the Telegram Bot
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by system/user.")
    except Exception as e:
        logger.critical(f"Critical failure: {e}")