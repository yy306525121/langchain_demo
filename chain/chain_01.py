from langchain import LlamaCpp, PromptTemplate, LLMChain

llm = LlamaCpp(model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf',
               max_tokens=1000,
               temperature=0.1,
               verbose=False,
               n_gpu_layers=0)

prompt = PromptTemplate(
    input_variables=["location"],
    template="我想去{location}旅行，我应该怎么办？",
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run('北京')
print(result)