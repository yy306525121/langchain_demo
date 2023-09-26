from langchain import LlamaCpp, LLMCheckerChain, LLMMathChain

chat = LlamaCpp(model_path='/Users/yangzy/Downloads/Llama2-chat-Chinese-50W/ggml-model-q4_0.gguf',
                max_tokens=1000,
                temperature=0.9,
                verbose=False,
                n_ctx=2048,
                n_gpu_layers=0)

llm_math = LLMMathChain.from_llm(llm=chat, verbose=True)
result = llm_math.run("13乘以3等于多少")
print(result)
