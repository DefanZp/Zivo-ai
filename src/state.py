from typing import TypedDict

from langchain.agents import AgentState

class ZivoAgentState(AgentState):
    selected_product_id: int | None

class ZivoAgentContext(TypedDict):
    cart_api_token: str | None