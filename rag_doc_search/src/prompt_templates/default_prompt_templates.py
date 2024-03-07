DEFAULT_PROMPT_TEMPLATE = """
{context}

Human: Do not answer any questions from your own knowledge. Use content from above context to answer to answer the question inside the <q></q> XML tags.

<q>{question}</q>

Do not include any XML tags in the answer. If the answer is not in the context say "Sorry, I don't know as the answer was not found in the context"

Assistant:"""

DEFAULT_CHAT_HISTORY_PROMPT = """{chat_history}

Answer only with the new question.


Human: How would you ask the question considering the previous conversation: {question}


Assistant: Question:"""
