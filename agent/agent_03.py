import os

from langchain import SerpAPIWrapper, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.tools import Tool

os.environ['SERPAPI_API_KEY'] = 'd2045544bb01461aceb438da968649d49e96faddcd0c8f0ef8478e344f284d45'
search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于从谷歌上搜索答案"
)


def fake_func(inp: str) -> str:
    return "foo"


fake_tools = [
    Tool(
        name=f"foo-{i}",
        func=fake_func,
        description=f"获取随机数 {i}"
    )
    for i in range(99)
]
ALL_TOOLS = [search_tool] + fake_tools

docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
vector_store = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name='/Users/yangzy/Downloads/text2vec-large-chinese'))

retriever = vector_store.as_retriever()


def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]


result = get_tools("天气怎么样")
print(result)
