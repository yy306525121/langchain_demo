from langchain import LlamaCpp, PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain

chat = LlamaCpp(model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf',
               max_tokens=1000,
               temperature=0.5,
               verbose=False,
               n_gpu_layers=0)

prompt = PromptTemplate(
    input_variables=['product_name'],
    template='为生产{product_name}的公司起一个公司名字'
)
# 定义一个chain,为产品取一个公司名称
chain = LLMChain(llm=chat, prompt=prompt)

second_prompt = PromptTemplate(
    input_variables=['company_name'],
    template='为{company_name}公司起一个口号'
)
# 定义第一个chain,为公司写一个口号
chain_two = LLMChain(llm=chat, prompt=second_prompt)

multi_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
result = multi_chain.run('泡泡口香糖')
print(result)
