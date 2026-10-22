from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)

messages = [
    HumanMessage(content="Halo, aku mencari hoodie."),
    AIMessage(content="Ada hoodie hitam yang tersedia."),
    HumanMessage(content="Berapa harganya?"),
    AIMessage(content="Harganya Rp150.000."),
    HumanMessage(content="Kalau warna putih ada?"),
    AIMessage(content="Ada, hoodie putih juga tersedia."),
]

tokens_before = count_tokens_approximately(messages)

trimmed_messages = trim_messages(
    messages,
    max_tokens=35,
    strategy="last",
    token_counter=count_tokens_approximately,
    start_on="human",
    include_system=True,
    allow_partial=False,
)

tokens_after = count_tokens_approximately(trimmed_messages)

print(f"Perkiraan token sebelum trimming: {tokens_before}")
print(f"Perkiraan token setelah trimming: {tokens_after}")
print("\nPesan setelah trimming:")

for message in trimmed_messages:
    print(f"{message.type}: {message.content}")