# YAIC IT Support RAG

RAG-асистент внутрішньої IT-підтримки Yet Another IT Company. Корпус містить
політики й інструкції у Markdown, заявки та каталог сервісів у CSV, а також договори
з постачальниками у PDF.

## Запуск

Потрібні Python 3.12 та [uv](https://docs.astral.sh/uv/).

1. Створіть `.env` за зразком `.env.example` і заповніть `OPENAI_API_KEY` та
   `TELEGRAM_BOT_TOKEN`. Токен Telegram видає `@BotFather` під час створення бота.
2. Встановіть залежності:

   ```shell
   uv sync --locked
   ```

3. Побудуйте локальний індекс ChromaDB:

   ```shell
   uv run python -m rag.index
   ```

4. Запустіть Telegram-бота:

   ```shell
   uv run python -m rag.bot
   ```

Бот відповідає українською лише за матеріалами корпусу та додає використані файли
у блоці `Джерела:`.
