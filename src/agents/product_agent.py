
import json

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from state import ZivoAgentContext, ZivoAgentState
from tools.agent_state_tools import (
    set_current_product,
    clear_current_product,
)


# ========================================
# PRODUCT AGENT PROMPT
# ========================================


PRODUCT_AGENT_PROMPT = """
You are Zivo's Product Specialist. Help users discover products,
verify product details, and select suitable products.

LANGUAGE:
- Respond in the user's language.
- Preserve official product names.

SOURCE OF TRUTH:
- Tool results are the source of truth.
- Product RAG provides candidates, not authoritative current data.
- Use get_product to verify product details, prices, and stock.
- Never invent product IDs or product information.

PRODUCT DISCOVERY:

1. ATTRIBUTE-BASED DISCOVERY

For requests based on materials, characteristics, features,
style, intended use, or combinations of attributes:

- Use search_product_knowledge with the user's relevant requirements.
- Evaluate candidates against those requirements.
- If one candidate is clearly the best match, call get_product
with its exact product ID.
- If multiple candidates are similarly suitable, return their
relevant details and IDs. Ask the user to choose.
- If no suitable candidate is found, use search_products
as a fallback when appropriate.

2. DIRECT CATALOG SEARCH

For requests based on product names, categories, or ordinary
catalog keywords:

- Use search_products.
- Convert the query into concise English catalog search terms.
- Preserve category, color, style, material, and other relevant
attributes.
- Translate the meaning, not just individual words.
- If multiple relevant products are found, let the user choose.

Examples:
- "Cari kaos" -> "t-shirt"
- "Cari kemeja linen" -> "linen shirt"
- "Cari hoodie hitam oversized" -> "oversized black hoodie"

PRODUCT VERIFICATION AND SELECTION:

- Use exact product IDs returned by tools or clearly identified
in previous conversation results.
- Before selecting a product, call get_product and verify that
it successfully returns the current product details.
- Only after successful verification, call set_current_product
with the exact product ID.
- Never select a product when the choice is ambiguous.
- If verification fails, do not update the current selection.
- Use clear_current_product when an existing selection must
be explicitly removed without a new product search.
- New product searches clear the previous selection.

SUPERVISOR HANDOFF:
- You are a specialized worker, not the main assistant.
- Return concise, factual findings.
- Include exact product IDs and relevant candidate details
when the user needs to choose.
- Include verified product information when available.
- Clearly state when no suitable product is found.
- Never invent missing information.

BOUNDARIES:
- Do not perform cart operations.
- Do not answer store policy questions.
- Do not treat RAG results as current prices or stock.

RESPONSE STYLE:
- Keep responses concise and factual.
- Explain why a product matches the user's requirements
using its documented characteristics.
- Include only relevant product details.
"""



# ========================================
# MCP PRODUCT TOOLS
# ========================================

def create_mcp_search_products_tool(mcp_search_tool):

    @tool("search_products")
    async def search_products(
        search: str,
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> Command:
        """
        Search the product catalog by product name,
        category, or keyword.

        Use concise English search terms.
        A new search clears the previous product selection.
        """

        result = await mcp_search_tool.ainvoke({
            "query": search,
            "limit": 5,
        })

        # Convert MCP result into text.
        if isinstance(result, str):
            content = result

        elif isinstance(result, list):
            text_parts = [
                item["text"]
                for item in result
                if isinstance(item, dict)
                and item.get("type") == "text"
            ]

            if text_parts:
                content = "\n".join(text_parts)
            else:
                content = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )

        else:
            content = json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

        # Clear previous product selection on new search.
        return Command(
            update={
                "selected_product_id": None,
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return search_products



# ========================================
# MCP PRODUCT KNOWLEDGE TOOL
# ========================================

def create_mcp_search_product_knowledge_tool(mcp_search_tool):

    @tool("search_product_knowledge")
    async def search_product_knowledge(
        query: str,
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> Command:
        """
        Find product candidates using semantic search.

        Use for material, characteristics, features, style,
        and intended use.

        Results are candidates, not verified product data.
        A new search clears the previous product selection.
        """

        result = await mcp_search_tool.ainvoke({
            "query": query,
        })

        # Convert MCP result into text.
        if isinstance(result, str):
            content = result

        elif isinstance(result, list):
            text_parts = [
                item["text"]
                for item in result
                if isinstance(item, dict)
                and item.get("type") == "text"
            ]

            if text_parts:
                content = "\n".join(text_parts)
            else:
                content = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )

        else:
            content = json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

        # Clear previous selection on new product search.
        return Command(
            update={
                "selected_product_id": None,
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return search_product_knowledge



# ========================================
# CREATE PRODUCT AGENT
# ========================================

def create_product_agent(model, mcp_tools):

    # Map available MCP tools by name.
    mcp_tools_by_name = {
        item.name: item
        for item in mcp_tools
    }

    required_tool_names = [
        "search_products",
        "get_product",
        "search_product_knowledge",
    ]

    missing_tools = [
        name
        for name in required_tool_names
        if name not in mcp_tools_by_name
    ]

    if missing_tools:
        raise RuntimeError(
            f"Missing MCP product tools: {', '.join(missing_tools)}"
        )

    # Create local wrapper for product search.
    search_products_tool = create_mcp_search_products_tool(
        mcp_tools_by_name["search_products"]
    )

    # Create local wrapper for product knowledge search.
    search_product_knowledge_tool = (
        create_mcp_search_product_knowledge_tool(
            mcp_tools_by_name["search_product_knowledge"]
        )
    )

    return create_agent(
        model=model,
        tools=[
            search_products_tool,
            mcp_tools_by_name["get_product"],
            search_product_knowledge_tool,
            set_current_product,
            clear_current_product,
        ],
        system_prompt=PRODUCT_AGENT_PROMPT,
        state_schema=ZivoAgentState,
    )


# ========================================
# PRODUCT AGENT TOOL FOR SUPERVISOR
# ========================================

def create_product_agent_tool(product_agent):

    @tool(
        "product_agent",
        description=(
            "Delegate product discovery, product search, "
            "product selection, and product detail tasks "
            "to the Product Agent."
        ),
    )
    async def call_product_agent(
        task: str,
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> Command:
        """
        Delegate product-related tasks to Product Agent.
        """

        current_product_id = ( runtime.state or {} ).get("selected_product_id")

        result = await product_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user", 
                        "content": task
                    }
                ],
                "selected_product_id": current_product_id,
            },
        )

        # Get the Product Agent's final response.
        messages = result.get("messages", [])

        if not messages:
            raise RuntimeError(
                "Product Agent returned no messages."
            )

        final_message = messages[-1]

        # Normalize the final response content.
        content = final_message.content

        if isinstance(content, str):
            response_text = content

        elif isinstance(content, list):
            response_text = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )

        else:
            response_text = ""

        if not response_text.strip():
            raise RuntimeError(
                "Product Agent returned empty response."
            )

        # Preserve the updated product selection, second argument is fallback.
        updated_product_id = result.get("selected_product_id", current_product_id,)

        # Return the response and state to Supervisor.
        return Command(
            update={
                "selected_product_id": updated_product_id,
                "messages": [
                    ToolMessage(
                        content=response_text,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return call_product_agent
