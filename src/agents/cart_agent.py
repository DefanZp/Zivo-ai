
import json

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

from state import ZivoAgentContext, ZivoAgentState


# ========================================
# CART AGENT PROMPT
# ========================================

CART_AGENT_PROMPT = """
You are Zivo's Cart Specialist.

Your responsibility is to help users view and manage
their shopping cart.

LANGUAGE:
- Respond in the user's language.
- Keep official product names in their original language.

IMPORTANT RULES:
- Use tool results as the source of truth.
- Never invent cart contents, product IDs, quantities,
prices, or operation results.
- Protected cart operations require authentication.
- Never bypass authentication or authorization.
- On AUTH_REQUIRED, tell the user to log in.
- On FORBIDDEN, explain that the action is unauthorized.
- Never claim an unsuccessful operation succeeded.

APPROVAL RULES:
- All cart mutation operations require explicit user approval.
- The mutation tools handle approval through LangGraph interrupts.
- Do not assume that the user's initial request is approval.
- Do not claim that an operation has succeeded while waiting
for approval.
- If an operation returns ACTION_REJECTED, inform the user
that the operation was not performed.
- After approval, let the interrupted tool execution continue.
- Do not call the same mutation tool again unnecessarily.
- Only report success after receiving a successful tool result.
- get_cart is read-only and does not require approval.

CART OPERATIONS:

For viewing the cart:
- Use get_cart to retrieve current cart contents.
- Report the result returned by the tool.
- If authentication is required, explain that the user
needs to log in.

For adding a product:
- add_to_cart uses selected_product_id from agent state.
- Use the quantity requested by the user.
- If no product is selected, do not invent a product ID
or search for products.
- Inform the Supervisor that a product must be selected
by the Product Agent before adding it.
- Check the tool result before confirming the addition.

For changing quantity:
- increase_quantity increases the selected cart item's
quantity by exactly one.
- decrease_quantity decreases the selected cart item's
quantity by exactly one.
- These tools do not accept a product ID.
- Do not call them if no product is selected.
- Check the tool result before confirming the change.

For removing a product:
- remove_from_cart removes the currently selected product.
- Do not call it if no product is selected.
- Check the tool result before confirming removal.

For clearing the cart:
- clear_cart removes all items from the user's cart.
- Check the tool result before confirming the operation.

GENERAL RULES:
- Never perform product discovery.
- Never invent or guess selected_product_id.
- Do not answer product discovery or store policy questions.
- Use only the tools relevant to the user's cart request.
- Return concise and factual findings to the Supervisor.

RESPONSE STYLE:
- Keep responses clear and concise.
- Include only relevant cart information.
- Never claim an operation succeeded without a successful tool result.
"""


# ========================================
# MCP RESULT CONVERSION
# ========================================

def format_mcp_result(result) -> str:

    if isinstance(result, str):
        return result

    if isinstance(result, list):
        text_parts = [
            item["text"]
            for item in result
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]

        if text_parts:
            return "\n".join(text_parts)

    return json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )


# ========================================
# MCP CART TOOLS
# ========================================

def create_mcp_get_cart_tool(mcp_get_cart_tool):

    @tool("get_cart")
    async def get_cart(
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> str:
        """
        Get the current authenticated user's shopping cart.
        """

        context = runtime.context or {}
        cart_api_token = context.get("cart_api_token")

        if (
            not isinstance(cart_api_token, str)
            or not cart_api_token.strip()
        ):
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": "The user must log in to view the cart.",
            })

        result = await mcp_get_cart_tool.ainvoke({
            "cart_api_token": cart_api_token,
        })

        return format_mcp_result(result)

    return get_cart


def create_mcp_add_to_cart_tool(mcp_add_to_cart_tool):

    @tool("add_to_cart")
    async def add_to_cart(
        quantity: int,
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> str:
        """Add the selected product to the user's cart."""

        state = runtime.state or {}
        product_id = state.get("selected_product_id")

        print(
            "[DEBUG add_to_cart] "
            f"selected_product_id: {product_id}"
        )

        if (
            isinstance(product_id, bool)
            or not isinstance(product_id, int)
            or product_id < 1
        ):
            return json.dumps({
                "success": False,
                "message": "No product is currently selected.",
            })

        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity < 1
        ):
            return json.dumps({
                "success": False,
                "message": "Quantity must be a positive integer.",
            })

        context = runtime.context or {}
        cart_api_token = context.get("cart_api_token")

        if (
            not isinstance(cart_api_token, str)
            or not cart_api_token.strip()
        ):
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": "Please log in before adding products.",
            })

        decision = interrupt({
            "type": "cart_action_approval",
            "message": "Review this action before adding the product.",
            "action": "add_to_cart",
            "arguments": {
                "product_id": product_id,
                "quantity": quantity,
            },
        })

        if (
            not isinstance(decision, dict)
            or decision.get("decision") != "approve"
        ):
            return json.dumps({
                "success": False,
                "code": "ACTION_REJECTED",
                "message": "The user rejected this action. No cart change was made.",
            })

        result = await mcp_add_to_cart_tool.ainvoke({
            "product_id": product_id,
            "quantity": quantity,
            "cart_api_token": cart_api_token,
        })

        return format_mcp_result(result)

    return add_to_cart


def create_mcp_increase_quantity_tool(mcp_increase_quantity_tool):

    @tool("increase_quantity")
    async def increase_quantity(
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> str:
        """Increase the selected cart item's quantity by one."""

        state = runtime.state or {}
        product_id = state.get("selected_product_id")

        if (
            isinstance(product_id, bool)
            or not isinstance(product_id, int)
            or product_id < 1
        ):
            return json.dumps({
                "success": False,
                "message": "No product is currently selected.",
            })

        context = runtime.context or {}
        cart_api_token = context.get("cart_api_token")

        if (
            not isinstance(cart_api_token, str)
            or not cart_api_token.strip()
        ):
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": "Please log in before updating the cart.",
            })

        decision = interrupt({
            "type": "cart_action_approval",
            "message": "Review this action before increasing the product quantity.",
            "action": "increase_quantity",
            "arguments": {
                "product_id": product_id,
                "quantity": 1,
            },
        })

        if (
            not isinstance(decision, dict)
            or decision.get("decision") != "approve"
        ):
            return json.dumps({
                "success": False,
                "code": "ACTION_REJECTED",
                "message": "The user rejected this action. No cart change was made.",
            })

        result = await mcp_increase_quantity_tool.ainvoke({
            "product_id": product_id,
            "cart_api_token": cart_api_token,
        })

        return format_mcp_result(result)

    return increase_quantity


def create_mcp_decrease_quantity_tool(mcp_decrease_quantity_tool):

    @tool("decrease_quantity")
    async def decrease_quantity(
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> str:
        """Decrease the selected cart item's quantity by one."""

        state = runtime.state or {}
        product_id = state.get("selected_product_id")

        if (
            isinstance(product_id, bool)
            or not isinstance(product_id, int)
            or product_id < 1
        ):
            return json.dumps({
                "success": False,
                "message": "No product is currently selected.",
            })

        context = runtime.context or {}
        cart_api_token = context.get("cart_api_token")

        if (
            not isinstance(cart_api_token, str)
            or not cart_api_token.strip()
        ):
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": "Please log in before updating the cart.",
            })

        decision = interrupt({
            "type": "cart_action_approval",
            "message": "Review this action before decreasing the product quantity.",
            "action": "decrease_quantity",
            "arguments": {
                "product_id": product_id,
                "quantity": 1,
            },
        })

        if (
            not isinstance(decision, dict)
            or decision.get("decision") != "approve"
        ):
            return json.dumps({
                "success": False,
                "code": "ACTION_REJECTED",
                "message": "The user rejected this action. No cart change was made.",
            })

        result = await mcp_decrease_quantity_tool.ainvoke({
            "product_id": product_id,
            "cart_api_token": cart_api_token,
        })

        return format_mcp_result(result)

    return decrease_quantity


def create_mcp_remove_from_cart_tool(mcp_remove_from_cart_tool):

    @tool("remove_from_cart")
    async def remove_from_cart(
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> str:
        """Remove the currently selected product from the cart."""

        state = runtime.state or {}
        product_id = state.get("selected_product_id")

        if (
            isinstance(product_id, bool)
            or not isinstance(product_id, int)
            or product_id < 1
        ):
            return json.dumps({
                "success": False,
                "message": "No valid product is currently selected.",
            })

        context = runtime.context or {}
        token = context.get("cart_api_token")

        if not isinstance(token, str) or not token.strip():
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": "Please log in before updating the cart.",
            })

        decision = interrupt({
            "type": "cart_action_approval",
            "message": "Review this action before removing the product from your cart.",
            "action": "remove_from_cart",
            "arguments": {
                "product_id": product_id,
            },
        })

        if (
            not isinstance(decision, dict)
            or decision.get("decision") != "approve"
        ):
            return json.dumps({
                "success": False,
                "code": "ACTION_REJECTED",
                "message": "The user rejected this action. No cart change was made.",
            })

        result = await mcp_remove_from_cart_tool.ainvoke({
            "product_id": product_id,
            "cart_api_token": token,
        })

        return format_mcp_result(result)

    return remove_from_cart


def create_mcp_clear_cart_tool(mcp_clear_cart_tool):

    @tool("clear_cart")
    async def clear_cart(
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> str:
        """Remove all items from the user's cart."""

        context = runtime.context or {}
        token = context.get("cart_api_token")

        if not isinstance(token, str) or not token.strip():
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": "Please log in before clearing the cart.",
            })

        decision = interrupt({
            "type": "cart_action_approval",
            "message": "Review this action before clearing your entire cart.",
            "action": "clear_cart",
        })

        if (
            not isinstance(decision, dict)
            or decision.get("decision") != "approve"
        ):
            return json.dumps({
                "success": False,
                "code": "ACTION_REJECTED",
                "message": "The user rejected this action. No cart change was made.",
            })

        result = await mcp_clear_cart_tool.ainvoke({
            "cart_api_token": token,
        })

        return format_mcp_result(result)

    return clear_cart


# ========================================
# CREATE CART AGENT
# ========================================

def create_cart_agent(model, mcp_tools):

    # Map available MCP tools by name.
    mcp_tools_by_name = {
        item.name: item
        for item in mcp_tools
    }

    required_tool_names = [
        "get_cart",
        "add_to_cart",
        "increase_quantity",
        "decrease_quantity",
        "remove_from_cart",
        "clear_cart",
    ]

    missing_tools = [
        name
        for name in required_tool_names
        if name not in mcp_tools_by_name
    ]

    if missing_tools:
        raise RuntimeError(
            f"Missing MCP cart tools: {', '.join(missing_tools)}"
        )

    # Create local wrappers for MCP cart tools.
    get_cart_tool = create_mcp_get_cart_tool(
        mcp_tools_by_name["get_cart"]
    )

    add_to_cart_tool = create_mcp_add_to_cart_tool(
        mcp_tools_by_name["add_to_cart"]
    )

    increase_quantity_tool = create_mcp_increase_quantity_tool(
        mcp_tools_by_name["increase_quantity"]
    )

    decrease_quantity_tool = create_mcp_decrease_quantity_tool(
        mcp_tools_by_name["decrease_quantity"]
    )

    remove_from_cart_tool = create_mcp_remove_from_cart_tool(
        mcp_tools_by_name["remove_from_cart"]
    )

    clear_cart_tool = create_mcp_clear_cart_tool(
        mcp_tools_by_name["clear_cart"]
    )

    return create_agent(
        model=model,
        tools=[
            get_cart_tool,
            add_to_cart_tool,
            increase_quantity_tool,
            decrease_quantity_tool,
            remove_from_cart_tool,
            clear_cart_tool,
        ],
        system_prompt=CART_AGENT_PROMPT,
        state_schema=ZivoAgentState,
        context_schema=ZivoAgentContext,
    )


# ========================================
# CART AGENT TOOL FOR SUPERVISOR
# ========================================

def create_cart_agent_tool(cart_agent):

    @tool(
        "cart_agent",
        description=(
            "Delegate shopping cart tasks such as viewing the cart, "
            "adding products, updating quantities, removing products, "
            "and clearing the cart to the Cart Agent."
        ),
    )
    async def call_cart_agent(
        task: str,
        runtime: ToolRuntime[ZivoAgentContext],
    ) -> Command:
        """
        Delegate cart-related tasks to Cart Agent.
        """

        current_product_id = (
            runtime.state or {}
        ).get("selected_product_id")

        print(
            "[DEBUG call_cart_agent] "
            f"selected_product_id: {current_product_id}"
        )
        
        result = await cart_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task,
                    }
                ],
                "selected_product_id": current_product_id,
            },
            context=runtime.context,
        )

        # Get the Cart Agent's final response.
        messages = result.get("messages", [])

        if not messages:
            raise RuntimeError(
                "Cart Agent returned no messages."
            )

        # Check whether the user rejected a cart operation.
        action_rejected = False

        for message in messages:

            if not isinstance(message, ToolMessage):
                continue

            if not isinstance(message.content, str):
                continue

            try:
                tool_result = json.loads(message.content)
            except json.JSONDecodeError:
                continue

            if (
                isinstance(tool_result, dict)
                and tool_result.get("code") == "ACTION_REJECTED"
            ):
                action_rejected = True
                break


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

        # Preserve the rejection result for Supervisor.
        if action_rejected:
            response_text = (
                "ACTION_REJECTED: The user explicitly rejected "
                "this operation. No cart change was made. "
                "Do not retry this operation."
            )

        if not response_text.strip():
            raise RuntimeError(
                "Cart Agent returned empty response."
            )

        # Preserve the selected product state.
        updated_product_id = result.get(
            "selected_product_id",
            current_product_id,
        )

        # Return response and state to Supervisor.
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

    return call_cart_agent
