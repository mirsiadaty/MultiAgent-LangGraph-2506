# MultiAgent-ViaLangGraph-2506
multi agent system MAS via LangGraph, using vendor-supplied LM

This example uses the "LangGraph" agent framework. LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. (https://langchain-ai.github.io/langgraph/)

This re-implements a LangGraph MAS (multi agent system) with three agents. We use OpenAI-supplied LM language model, than a local LM. (https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb)

The following shows the overall architecture of this MAS:
* supervisor agent
* math agent
* research agent

![lg34](https://github.com/user-attachments/assets/2052f46b-8241-49a1-91c5-df55c1ebeb68)

<br>
<br>

## Building the agents

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


The following is the final response from this MAS for the question "find US and New York state GDP in 2024. what % of US GDP was New York state?":

```
================================== Ai Message ==================================
Name: supervisor

In 2024:
- US GDP is estimated at $22.673 trillion.
- New York state GDP is estimated at $2.297 trillion.
- New York state's GDP is approximately 10.13% of the US GDP.
99 Tue Jun 24 16:32:52 202
```

The following is the detailed messaging history between the three agents, collaboratively solving the given question:

```
================================ Human Message =================================

find US and New York state GDP in 2024. what % of US GDP was New York state?
================================== Ai Message ==================================
Name: supervisor
Tool Calls:
  transfer_to_research_agent (call_PhUOzbsZISA4YjBB2Tr9jNXf)
 Call ID: call_PhUOzbsZISA4YjBB2Tr9jNXf
  Args:
================================= Tool Message =================================
Name: transfer_to_research_agent

Successfully transferred to research_agent
================================== Ai Message ==================================
Name: research_agent
Tool Calls:
  tavily_search (call_l9DNLdkSNfVR91XF3Ds6R4tn)
 Call ID: call_l9DNLdkSNfVR91XF3Ds6R4tn
  Args:
    query: US GDP 2024 estimate
    search_depth: advanced
  tavily_search (call_JjPMrf7fDvXLjaifRyTvbxqO)
 Call ID: call_JjPMrf7fDvXLjaifRyTvbxqO
  Args:
    query: New York state GDP 2024 estimate
    search_depth: advanced
================================= Tool Message =================================
Name: tavily_search

{"query": "US GDP 2024 estimate", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.worldeconomics.com/GrossDomesticProduct/Real-GDP/United%20States.aspx", "title": "United States GDP: $22.673 Trillion - World Economics", "content": "United States GDP: $22.673 Trillion United States GDP Forecast: $22.673 Trillion in 2024, $23.285 Trillion projected for 2025 (Constant 2015 Prices). GDP in United States is estimated to be $22.673 Trillion US dollars at the end of 2024 in Real GDP terms. Looking ahead to 2025, projections suggest United States's 2025 GDP estimate could be $23.285 Trillion. This United States GDP growth forecast for 2024 and 2025 reflects an estimated growth rate of 2.7%. Real GDP (US$): United States Real GDP per Capita(US$, 2010) | 62,996 | 64,342 | 65,875 | 67,324 GDP per Capita Annual Growth Rate(%) | 5.9 | 2.1 | 2.4 | 2.2 Data Quality Rating for United States | See how much you can trust United States's GDP Data", "score": 0.9469168, "raw_content": null}, {"url": "https://www.statista.com/statistics/216985/forecast-of-us-gross-domestic-product/", "title": "GDP forecast U.S. 2034 - Statista", "content": "[](https://www.statista.com/statistics/216985/forecast-of-us-gross-domestic-product/#statisticContainer) This graph shows a forecast of the gross domestic product of the United States of America for fiscal years 2024 to 2034. GDP refers to the market value of all final goods and services produced within a country in a given period. According to the CBO, the United States GDP will increase steadily over the next decade from 28.18 trillion U.S. dollars in 2023 to 41.65 trillion U.S. dollars in [...] US Congressional Budget Office. (February 7, 2024). Forecast of the gross domestic product of the United States from fiscal year 2024 to fiscal year 2034 (in billion U.S. dollars) [Graph]. In Statista. Retrieved May 20, 2025, from https://www.statista.com/statistics/216985/forecast-of-us-gross-domestic-product/ [...] US Congressional Budget Office. \"Forecast of The Gross Domestic Product of The United States from Fiscal Year 2024 to Fiscal Year 2034 (in Billion U.S. Dollars).\" Statista, Statista Inc., 7 Feb 2024, https://www.statista.com/statistics/216985/forecast-of-us-gross-domestic-product/", "score": 0.8747745, "raw_content": null}, {"url": "https://tradingeconomics.com/united-states/gdp-growth", "title": "United States GDP Growth Rate - Trading Economics", "content": "| [GDP from Services](https://tradingeconomics.com/united-states/gdp-from-services) | 17050.50 | 16949.30 | USD Billion | Dec 2024 |\n| [GDP from Transport](https://tradingeconomics.com/united-states/gdp-from-transport) | 730.50 | 721.40 | USD Billion | Dec 2024 |\n| [GDP from Utilities](https://tradingeconomics.com/united-states/gdp-from-utilities) | 350.80 | 341.40 | USD Billion | Dec 2024 | [...] | [GDP from Manufacturing](https://tradingeconomics.com/united-states/gdp-from-manufacturing) | 2406.80 | 2402.80 | USD Billion | Dec 2024 |\n| [GDP from Mining](https://tradingeconomics.com/united-states/gdp-from-mining) | 343.60 | 337.60 | USD Billion | Dec 2024 |\n| [GDP from Public Administration](https://tradingeconomics.com/united-states/gdp-from-public-administration) | 2653.10 | 2635.50 | USD Billion | Dec 2024 | [...] | [Gross Fixed Capital Formation](https://tradingeconomics.com/united-states/gross-fixed-capital-formation) | 4346.50 | 4265.90 | USD Billion | Mar 2025 |\n| [Gross National Product](https://tradingeconomics.com/united-states/gross-national-product) | 23620.90 | 23427.70 | USD Billion | Dec 2024 |\n| [Real Consumer Spending](https://tradingeconomics.com/united-states/real-consumer-spending) | 1.20 | 4.00 | percent | Mar 2025 |", "score": 0.8315952, "raw_content": null}], "response_time": 2.11}
================================= Tool Message =================================
Name: tavily_search

{"query": "New York state GDP 2024 estimate", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://en.wikipedia.org/wiki/Economy_of_New_York_(state)", "title": "Economy of New York (state) - Wikipedia", "content": "The **economy of the State of New York** is reflected in its [gross state product](https://en.wikipedia.org/wiki/Gross_state_product \"Gross state product\") in 2024 of $2.297 trillion, ranking [third](https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP \"List of U.S. states and territories by GDP\") in size behind the larger states of [California](https://en.wikipedia.org/wiki/Economy_of_California \"Economy of California\") and [...] world</a></div></td></tr><tr><th colspan=\"2\" class=\"infobox-header\" style=\"background:lightblue;\">Statistics</th></tr><tr><th scope=\"row\" class=\"infobox-label\"><a href=\"/wiki/Gross_domestic_product\" title=\"Gross domestic product\">GDP</a></th><td class=\"infobox-data\">$2.297 trillion (2024)<sup id=\"cite_ref-GDPByState_1-0\" class=\"reference\"><a href=\"#cite_note-GDPByState-1\"><span class=\"cite-bracket\">[</span>1<span class=\"cite-bracket\">]</span></a></sup></td></tr><tr><th scope=\"row\" [...] class=\"infobox-label\"><div style=\"display: inline-block; line-height: 1.2em; padding: .1em 0;\">GDP per capita</div></th><td class=\"infobox-data\">$115,619 (2024)<sup id=\"cite_ref-PopulationDataSource_2-0\" class=\"reference\"><a href=\"#cite_note-PopulationDataSource-2\"><span class=\"cite-bracket\">[</span>2<span class=\"cite-bracket\">]</span></a></sup></td></tr><tr><th scope=\"row\" class=\"infobox-label\"><div style=\"display: inline-block; line-height: 1.2em; padding: .1em 0;\">Population below <span", "score": 0.92343384, "raw_content": null}, {"url": "https://comptroller.nyc.gov/reports/annual-state-of-the-citys-economy-and-finances-2024/", "title": "Annual State of the City's Economy and Finances 2024", "content": "[[1]](https://comptroller.nyc.gov/reports/annual-state-of-the-citys-economy-and-finances-2024/#_ftnref1) At that time, the Comptroller’s Office expected five-year cumulative real GDP growth, 2020 to 2024, of 8.7 percent while the mayor expected 9.5 percent.\n\n[[2]](https://comptroller.nyc.gov/reports/annual-state-of-the-citys-economy-and-finances-2024/#_ftnref2) U.S. Bureau of Labor Statistics Local Area Unemployment Statistics (LAUS). [...] (constant 2019 dollars)**22,550 15,322 13,596 19,451 21,211 21,096\n**_% change_**(32.1%)(11.3%)43.1%9.0%(0.5%)\n\nSource: NY State Department of Taxation and Finance and Office of the New York City Comptroller [...] | Retail Trade | 301,565 | $56,208 | 14.4 | (0.6) | (5.2) | 2.7 | 2.9 | 2.2 | 2.1 |\n| Leisure & Hospitality | 431,227 | $56,086 | 96.2 | 32.4 | 15.4 | 8.9 | 3.8 | 4.5 | 7.3 |\n| **Total: Low Wage Sectors** | **1,997,491** | **$62,185** | **184.6** | **102.2** | **91.6** | **62.7** | **32.0** | **26.8** | **29.3** |\n| **Total NYC Employment** | **4,485,695** | **$114,294** | **307.7** | **119.9** | **78.1** | **88.0** | **65.1** | **53.2** | **51.4** |", "score": 0.8660535, "raw_content": null}], "response_time": 2.07}
================================== Ai Message ==================================
Name: research_agent

- US GDP for 2024 is estimated to be $22.673 trillion (constant 2015 dollars) [Source: World Economics](https://www.worldeconomics.com/GrossDomesticProduct/Real-GDP/United%20States.aspx).
- New York state GDP for 2024 is estimated to be $2.297 trillion [Source: Wikipedia](https://en.wikipedia.org/wiki/Economy_of_New_York_(state)).
- New York state's GDP is approximately 10.1% of the US GDP in 2024.
================================== Ai Message ==================================
Name: research_agent

Transferring back to supervisor
Tool Calls:
  transfer_back_to_supervisor (a26fa2b5-950c-4e67-bff1-a274d979bcc2)
 Call ID: a26fa2b5-950c-4e67-bff1-a274d979bcc2
  Args:
================================= Tool Message =================================
Name: transfer_back_to_supervisor

Successfully transferred back to supervisor
================================== Ai Message ==================================
Name: supervisor
Tool Calls:
  transfer_to_math_agent (call_gqeM2UMXHh5yH7ZLbMyOpA61)
 Call ID: call_gqeM2UMXHh5yH7ZLbMyOpA61
  Args:
================================= Tool Message =================================
Name: transfer_to_math_agent

Successfully transferred to math_agent
================================== Ai Message ==================================
Name: math_agent
Tool Calls:
  divide (call_Dr6M6ZiRCrcwFZjxn1ndCISY)
 Call ID: call_Dr6M6ZiRCrcwFZjxn1ndCISY
  Args:
    a: 2.297
    b: 22.673
================================= Tool Message =================================
Name: divide

0.10130992810832269
================================== Ai Message ==================================
Name: math_agent

US GDP in 2024: $22.673 trillion  
New York state GDP in 2024: $2.297 trillion  
Percentage of US GDP that is New York state: 10.13%
================================== Ai Message ==================================
Name: math_agent

Transferring back to supervisor
Tool Calls:
  transfer_back_to_supervisor (32dfb7c2-a2ce-4406-a575-b3a5a46bebbb)
 Call ID: 32dfb7c2-a2ce-4406-a575-b3a5a46bebbb
  Args:
================================= Tool Message =================================
Name: transfer_back_to_supervisor

Successfully transferred back to supervisor
================================== Ai Message ==================================
Name: supervisor

In 2024:
- US GDP is estimated at $22.673 trillion.
- New York state GDP is estimated at $2.297 trillion.
- New York state's GDP is approximately 10.13% of the US GDP.
99 Tue Jun 24 16:32:52 202
```





