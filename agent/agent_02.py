import os
from typing import List, Tuple, Any, Union

from langchain import SerpAPIWrapper
from langchain.agents import BaseSingleActionAgent, AgentExecutor
from langchain.callbacks.base import Callbacks
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool

# 自定义代理演示
class FakeAgent(BaseSingleActionAgent):
    @property
    def input_keys(self) -> List[str]:
        return ['input']

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks = None, **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        return AgentAction(tool='Search', tool_input=kwargs['input'], log='')

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks = None, **kwargs: Any) -> Union[
        AgentAction, AgentFinish]:
        return AgentAction(tool='Search', tool_input=kwargs['input'], log='')

os.environ['SERPAPI_API_KEY'] = 'd2045544bb01461aceb438da968649d49e96faddcd0c8f0ef8478e344f284d45'
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="用于回答问题",
        return_direct=True
    )
]
agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
result = agent_executor.run('截止2023年，中国一共有多少人口？')
print(result)