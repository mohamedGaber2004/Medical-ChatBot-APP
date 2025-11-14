# memory_store.py
from langchain_community.chat_message_histories import ChatMessageHistory

# store per-session ChatMessageHistory objects
store = {}

def get_memory(session_id="user"):
    """
    Return a ChatMessageHistory object for the given session_id.
    RunnableWithMessageHistory expects the returned object to expose `.messages`.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
