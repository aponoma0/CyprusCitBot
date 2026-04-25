import os
import threading
import logging
from flask import Flask
from huggingface_hub import InferenceClient

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

# ======================
# LOGGING
# ======================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ======================
# FLASK (keep alive HF Space)
# ======================
server = Flask(__name__)

@server.route('/')
def health():
    return "Cyprus Citizenship Bot is Running 🇨🇾"

def run_flask():
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

# ======================
# SYSTEM PROMPT (CYPRUS EXAM)
# ======================
SYSTEM_PROMPT = """
You are a helpful tutor for the Cyprus citizenship exam.

RULES:
- Explain simply and clearly
- Use English or Russian depending on user language
- Focus on: history, geography, government, culture of Cyprus
- If user is confused, simplify explanation
- Be short and useful, like a teacher
"""

# ======================
# AI FUNCTION
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
        logging.error(f"AI error: {e}")
        return "Sorry, AI is not available right now. Try again."

# ======================
# START COMMAND
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🇨🇾 Cyprus Citizenship Helper Bot\n\n"
        "Ask me anything about Cyprus citizenship exam."
    )

# ======================
# MAIN MESSAGE HANDLER
# ======================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    response = query_ai(user_text)

    await update.message.reply_text(response)

# ======================
# MAIN APP
# ======================
def main():
    # start flask in background
    threading.Thread(target=run_flask, daemon=True).start()

    # telegram bot
    app = ApplicationBuilder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Cyprus bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()