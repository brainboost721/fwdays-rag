import asyncio
import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag.answer import rag_answer_with_router

WELCOME_MESSAGE = (
    "Вітаю! Я асистент внутрішньої IT-підтримки YAIC. "
    "Поставте запитання про наші політики, сервіси, заявки або договори."
)
ERROR_MESSAGE = "Не вдалося підготувати відповідь. Спробуйте ще раз пізніше."

logger = logging.getLogger(__name__)


async def start(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(WELCOME_MESSAGE)


async def answer_message(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text or not message.text.strip():
        return

    await message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await asyncio.to_thread(
            rag_answer_with_router, message.text.strip()
        )
    except Exception:
        logger.exception("Failed to answer Telegram message")
        await message.reply_text(ERROR_MESSAGE)
        return

    await message.reply_text(answer)


def build_application(token: str) -> Application:
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, answer_message)
    )
    return application


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Немає TELEGRAM_BOT_TOKEN — додайте його у .env")

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_application(token).run_polling()


if __name__ == "__main__":
    main()
