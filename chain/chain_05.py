from langchain import LlamaCpp, LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

chat = LlamaCpp(model_path='/Users/yangzy/Downloads/Llama2-chat-Chinese-50W/ggml-model-q4_0.gguf',
                max_tokens=1000,
                temperature=0.5,
                verbose=False,
                n_gpu_layers=0)

template = """你是一个乐于助人的助手，会生成逗号分隔的列表。
用户将输入一个类别，你应在逗号分隔的列表中生成该类别中的 5 个对象。
只返回一个逗号分隔的列表，仅此而已"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt, output_parser=CommaSeparatedListOutputParser())
result = chain.run('水果')
print(result)
