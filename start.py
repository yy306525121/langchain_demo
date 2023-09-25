from langchain import PromptTemplate, LlamaCpp, LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory

llm = LlamaCpp(model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf',
               max_tokens=1000,
               temperature=0.1,
               verbose=False,
               n_gpu_layers=0)

template = """我想去{location}旅行，我应该怎么办"""
prompt = PromptTemplate(
    input_variables=["location"], template=template
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
prompt = prompt.format(location='北京')

result = llm_chain.run(prompt)
print(result)
