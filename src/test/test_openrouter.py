import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(
    model=os.getenv("OPENROUTER_MODEL"),
    temperature=0,
)


response = model.invoke("Halo, perkenalkan dirimu dalam satu kalimat.")

print(response.content)