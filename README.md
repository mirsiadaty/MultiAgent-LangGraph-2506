# MultiAgent-ViaLangGraph-2506
multi agent system MAS via LangGraph and using vendor-supplied LM

This example uses the "LangGraph" agent framework. LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. (https://langchain-ai.github.io/langgraph/)

This re-implements a LangGraph MAS (multi agent system) with three agents. We use OpenAI-supplied LM language model, than a local LM. (https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb)

The following shows the overall architecture of this MAS:
* supervisor agent
* match agent
* research agent

![lg34](https://github.com/user-attachments/assets/2052f46b-8241-49a1-91c5-df55c1ebeb68)



