from langchain import LlamaCpp, LLMCheckerChain

chat = LlamaCpp(model_path='/Users/yangzy/Downloads/Llama2-chat-Chinese-50W/ggml-model-q4_0.gguf',
                max_tokens=1000,
                temperature=0.9,
                verbose=False,
                n_ctx=2048,
                n_gpu_layers=0)

text = "哪种动物产的蛋最大？"
checker_chain = LLMCheckerChain.from_llm(llm=chat, verbose=True)
result = checker_chain.run(text)
print(result)