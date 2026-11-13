import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
# from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.cart_agent import create_cart_agent, create_cart_agent_tool
from agents.knowledge_agent import create_knowledge_agent, create_knowledge_agent_tool
from agents.product_agent import create_product_agent, create_product_agent_tool
from state import ZivoAgentContext, ZivoAgentState


load_dotenv()

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0,
    timeout=180,
    max_retries=1,
)

summarization = SummarizationMiddleware(
    model=model,
    trigger=("tokens", 4000),
    keep=("messages", 6)
)




SUPERVISOR_PROMPT = """
You are Zivo's main shopping assistant. Coordinate specialized agents
and provide the final response.

LANGUAGE:
- Respond in the user's language.
- Preserve official product names.

AVAILABLE AGENTS:
- product_agent: Product discovery, search, details, and selection.
- cart_agent: Cart viewing and cart operations.
- knowledge_agent: Return, refund, shipping, and store policies.

ROUTING:
- Delegate product requests to product_agent.
- Delegate cart requests to cart_agent.
- Delegate store policy questions to knowledge_agent.
- Never access backend MCP tools directly.
- Never invent product IDs, prices, stock, cart contents, or policies.

PRODUCT AND CART WORKFLOW:
- For a cart operation, determine which product the user intends.
- If selected_product_id matches that product, delegate directly
to cart_agent.
- Otherwise, delegate product discovery and verification to
product_agent first.
- Wait for product_agent to finish before calling cart_agent.
Never run dependent product and cart tasks in parallel.
- Proceed with the cart operation only when one suitable product
is clearly identified.
- If multiple candidates are similarly suitable or the intended
product is ambiguous, ask the user to choose.
- When referring to previous search results, provide the relevant
candidate information and exact product IDs to product_agent.

CONTEXT AND STATE:
- You own the main conversation history.
- Specialized agents are stateless between invocations.
- Include relevant conversation context, previous candidates,
and product IDs when delegating tasks.
- Never assume a worker remembers previous invocations.

AUTHENTICATION:
- Protected cart operations require authentication.
- Follow authentication and authorization results from cart_agent.
- Never bypass access controls or claim an unsuccessful operation
succeeded.

FINAL RESPONSE:
- Use worker results as the source of truth.
- Do not expose internal agent operations.
- Keep responses concise, natural, and factual.
- Include only information relevant to the request.
"""


# fungsi untuk create supervisor agent
def create_zivo_agent(checkpointer, mcp_tools):

    # Create specialized agents.
    product_agent = create_product_agent(
        model,
        mcp_tools
    )

    cart_agent = create_cart_agent(
        model,
        mcp_tools
    )

    knowledge_agent = create_knowledge_agent(
        model,
        mcp_tools
    )

    # Wrap specialized agents as Supervisor tools.
    product_agent_tool = create_product_agent_tool(
        product_agent
    )

    cart_agent_tool = create_cart_agent_tool(
        cart_agent
    )

    knowledge_agent_tool = create_knowledge_agent_tool(
        knowledge_agent
    )

    # Create Supervisor
    return create_agent(
        model=model,
        tools=[
            product_agent_tool,
            cart_agent_tool,
            knowledge_agent_tool
        ],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[
            summarization,
        ],
        checkpointer=checkpointer,
        state_schema=ZivoAgentState,
        context_schema=ZivoAgentContext,
        name="zivo_supervisor",
    )


