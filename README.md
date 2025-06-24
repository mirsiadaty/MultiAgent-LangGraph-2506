# MultiAgent-ViaLangGraph-2506
multi agent system MAS via LangGraph, using vendor-supplied LM

This example uses the "LangGraph" agent framework. LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. (https://langchain-ai.github.io/langgraph/)

This re-implements a LangGraph MAS (multi agent system) with three agents. We use OpenAI-supplied LM language model, than a local LM. (https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb)

The following shows the overall architecture of this MAS:
* supervisor agent
* match agent
* research agent

![lg34](https://github.com/user-attachments/assets/2052f46b-8241-49a1-91c5-df55c1ebeb68)

The following excerpt shows code to build the three agents listed above.

```
math_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

research_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

```




