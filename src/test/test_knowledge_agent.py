
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage

import agents.knowledge_agent as knowledge_module

from agents.knowledge_agent import (
    create_knowledge_agent,
    create_knowledge_agent_tool,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ========================================
# MOCK OBJECTS
# ========================================

class FakeKnowledgeAgent:

    def __init__(self, result):
        self.result = result
        self.received_input = None

    async def ainvoke(self, input):
        self.received_input = input
        return self.result


class FakeMCPTool:

    def __init__(self, name):
        self.name = name


# ========================================
# KNOWLEDGE AGENT CREATION TESTS
# ========================================

def test_knowledge_agent_requires_policy_tool():

    with pytest.raises(
        RuntimeError,
        match="search_policy_knowledge",
    ):
        create_knowledge_agent(
            model=None,
            mcp_tools=[],
        )


def test_knowledge_agent_registers_policy_tool(monkeypatch):

    policy_tool = FakeMCPTool(
        "search_policy_knowledge"
    )

    expected_agent = object()
    captured_arguments = {}

    def fake_create_agent(**kwargs):
        captured_arguments.update(kwargs)
        return expected_agent

    monkeypatch.setattr(
        knowledge_module,
        "create_agent",
        fake_create_agent,
    )

    result = create_knowledge_agent(
        model="fake-model",
        mcp_tools=[policy_tool],
    )

    assert result is expected_agent
    assert captured_arguments["model"] == "fake-model"
    assert captured_arguments["tools"] == [policy_tool]
    assert (
        captured_arguments["system_prompt"]
        == knowledge_module.KNOWLEDGE_AGENT_PROMPT
    )


# ========================================
# KNOWLEDGE AGENT TOOL TESTS
# ========================================

@pytest.mark.anyio
async def test_knowledge_agent_receives_task():

    knowledge_agent = FakeKnowledgeAgent({
        "messages": [
            AIMessage(
                content="Our return policy allows returns within 7 days."
            )
        ]
    })

    knowledge_tool = create_knowledge_agent_tool(
        knowledge_agent
    )

    result = await knowledge_tool.coroutine(
        task="What is the return policy?"
    )

    assert knowledge_agent.received_input == {
        "messages": [
            {
                "role": "user",
                "content": "What is the return policy?",
            }
        ]
    }

    assert result == (
        "Our return policy allows returns within 7 days."
    )


@pytest.mark.anyio
async def test_knowledge_agent_normalizes_list_content():

    knowledge_agent = FakeKnowledgeAgent({
        "messages": [
            AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Returns are accepted within 7 days.",
                    },
                    {
                        "type": "text",
                        "text": "The product must be unused.",
                    },
                ]
            )
        ]
    })

    knowledge_tool = create_knowledge_agent_tool(
        knowledge_agent
    )

    result = await knowledge_tool.coroutine(
        task="Explain the return policy."
    )

    assert result == (
        "Returns are accepted within 7 days.\n"
        "The product must be unused."
    )


@pytest.mark.anyio
async def test_knowledge_agent_rejects_empty_messages():

    knowledge_agent = FakeKnowledgeAgent({
        "messages": [],
    })

    knowledge_tool = create_knowledge_agent_tool(
        knowledge_agent
    )

    with pytest.raises(
        RuntimeError,
        match="Knowledge Agent returned no messages",
    ):
        await knowledge_tool.coroutine(
            task="What is the shipping policy?"
        )


@pytest.mark.anyio
async def test_knowledge_agent_rejects_empty_response():

    knowledge_agent = FakeKnowledgeAgent({
        "messages": [
            AIMessage(content="   ")
        ]
    })

    knowledge_tool = create_knowledge_agent_tool(
        knowledge_agent
    )

    with pytest.raises(
        RuntimeError,
        match="Knowledge Agent returned empty response",
    ):
        await knowledge_tool.coroutine(
            task="What is the return policy?"
        )
