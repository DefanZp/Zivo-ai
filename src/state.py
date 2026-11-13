from typing import TypedDict
from typing_extensions import NotRequired

from langchain.agents import AgentState

class ZivoAgentState(AgentState):
    selected_product_id: NotRequired[int | None]

class ZivoAgentContext(TypedDict):
    cart_api_token: str | None