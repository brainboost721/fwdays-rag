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

   Повторний запуск оновлює наявні чанки завдяки стабільним ID. Щоб також видалити
   застарілі чанки після видалення файлів або зміни chunking, перебудуйте індекс:

   ```shell
   uv run python -m rag.index --rebuild
   ```

4. Запустіть Telegram-бота:

   ```shell
   uv run python -m rag.bot
   ```

Бот відповідає українською лише за матеріалами корпусу та додає використані файли
у блоці `Джерела:`.

## Evaluation

Golden set із трьома перевірочними питаннями зберігається у
`evaluation/golden_set.json`. Автоматична evaluation перевіряє точність джерел,
очікувану відмову для питання без відповіді та відповідність змісту еталону через
LLM-judge:

```shell
uv run python -m rag.evaluate
```

Команда друкує результат кожного кейса, source recall/precision та загальний score.
Під час запуску вона звертається до OpenAI для генерації відповідей і LLM-judge.
Якщо хоча б один кейс не пройшов перевірку, процес завершується з кодом `1`.
