import os

from langchain import LlamaCpp
from langchain.agents import load_tools, initialize_agent, AgentType

# chat = LlamaCpp(model_path='/Users/yangzy/Downloads/Llama2-chat-Chinese-50W/ggml-model-q4_0.gguf',
#                max_tokens=1000,
#                temperature=0.5,
#                verbose=False,
#                n_gpu_layers=0)
#
# os.environ['SERPAPI_API_KEY'] = 'd2045544bb01461aceb438da968649d49e96faddcd0c8f0ef8478e344f284d45'
# tools = load_tools(["serpapi", "llm-math"], llm=chat)
# agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# result = agent.run("谁是里奥-迪卡普里奥的女朋友？她现在的年龄是多少？")
# print(result)

