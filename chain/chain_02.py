from langchain import LlamaCpp, PromptTemplate, LLMChain

chat = LlamaCpp(model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf',
               max_tokens=1000,
               temperature=0.9,
               verbose=False,
               n_gpu_layers=0)

promptTemplate = '我想去{location}旅行，我应该怎么办？'
chain = LLMChain(llm=chat,
                 prompt=PromptTemplate.from_template(promptTemplate))
# result = chain(inputs={'location', '北京'}, return_only_outputs=False)
result = chain.run('北京')
print(result)