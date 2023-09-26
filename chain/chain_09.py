from langchain import LlamaCpp, PromptTemplate, LLMChain
from langchain.chains import LLMRequestsChain

chat = LlamaCpp(model_path='/Users/yangzy/Downloads/Llama2-chat-Chinese-50W/ggml-model-q4_0.gguf',
                max_tokens=1000,
                temperature=0.9,
                verbose=False,
                n_ctx=2048,
                n_gpu_layers=0)

template = """>>> 和 <<< 是谷歌搜索到的结果.
提取'{query}'的问题，并查找答案，如果没有找到答案返回'not found'.
使用下面的模板回答问题
Extracted:<答案 或 "not found">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)
chain = LLMRequestsChain(llm_chain = LLMChain(llm=chat, prompt=PROMPT))
question = "地球上最大的3个国家和他们的面积是多少？"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}
result = chain(inputs)
print(result)