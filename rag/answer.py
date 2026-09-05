import re

from rag.config import CHAT_MODEL, TOP_K
from rag.embeddings import get_client
from rag.retrieve import Hit, retrieve

NO_DATA = "У наших матеріалах немає даних для відповіді на це питання."

PROMPT_TEMPLATE = """Ти внутрішній асистент IT-підтримки компанії Yet Another IT Company.
Відповідай українською.

Правила:
- Спирайся ЛИШЕ на наведений контекст. Не додавай нічого із загальних знань.
- Якщо в контексті немає даних для відповіді — напиши рівно: «{no_data}»
  і не додавай блок «Джерела:». Краще чесно відмовити, ніж здогадуватись.
- Не вигадуй цифр, строків, сум, назв систем і номерів заявок, яких немає в контексті.
- У кінці додай блок «Джерела:» — маркований список рядків точно у форматі заголовків
  контексту: [source=…].
- Перелічуй лише ті [source=…], на які справді спирається відповідь, без повторів.
- Джерелом завжди є шлях до файлу (наприклад data/tickets.csv), а не внутрішні поля
  з тексту фрагмента (ticket_id, service_id тощо).

Контекст:
{context}

Питання: {question}
Відповідь:"""

SOURCE_RE = re.compile(r"\[source=([^\]]+)\]")


def _is_refusal(answer: str) -> bool:
    return answer.strip(" «\"'*").startswith(NO_DATA)


def format_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(f"[source={h['source']}]\n{h['document']}" for h in hits)


def drop_unseen_sources(answer: str, allowed: set[str]) -> str:
    """Remove [source=…] citations pointing at files that never reached the context."""
    kept = []
    for line in answer.splitlines():
        if SOURCE_RE.search(line):
            line = SOURCE_RE.sub(lambda m: m.group(0) if m.group(1) in allowed else "", line)
            if not line.strip(" -*\t"):
                continue
        kept.append(line)
    return "\n".join(kept)


def rag_answer(question: str, k: int = TOP_K, where: dict | None = None) -> str:
    hits = retrieve(question, k=k, where=where)
    if not hits:
        return NO_DATA

    prompt = PROMPT_TEMPLATE.format(
        no_data=NO_DATA, context=format_context(hits), question=question
    )
    completion = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = (completion.choices[0].message.content or "").strip()

    # The model keeps appending a sources block to refusals despite the prompt saying not to.
    if _is_refusal(answer):
        return NO_DATA

    return drop_unseen_sources(answer, {h["source"] for h in hits})
