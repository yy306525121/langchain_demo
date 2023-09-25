from typing import List, Dict, Any, Optional

from langchain import LlamaCpp, PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import SimpleSequentialChain
from langchain.chains.base import Chain


class CustomChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}


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
    input_variables=['product_name'],
    template='为生产{product_name}的公司起一个口号'
)
# 定义第一个chain,为公司写一个口号
chain_two = LLMChain(llm=chat, prompt=second_prompt)

multi_chain = CustomChain(chain_1=chain, chain_2=chain_two, verbose=True)
result = multi_chain.run('运动鞋')
print(result)
