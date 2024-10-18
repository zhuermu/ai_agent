import config
from hunyuan import HunyuanEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from typing import List
from langchain_core.documents import Document

## 1. 混元embedding 初始化
secretid = config.get_settings().tcloud_secret_id
secretkey = config.get_settings().tcloud_secret_key
## 1.1 注意这里HunyuanEmbeddings在我附件提供的github仓库中，需要自行下载
embeddings = HunyuanEmbeddings(
    secret_id=secretid,
    secret_key=secretkey,
    embedding_ctx_length=786,
)

## 2. 初始化elasticsearch store
ES_URL = config.get_settings().es_url
ES_API_KEY = config.get_settings().es_api_key

elastic_vector_search = ElasticsearchStore(
    embedding=embeddings,
    index_name="langchain_index",
    es_url=ES_URL,
    es_api_key=ES_API_KEY,
)

documents: List[Document] = []
text = '''
# 2024年巴黎奥运会的金牌情况

## 1. 2024巴黎奥运会概况：
   - **赛事时间与项目数量**：2024年巴黎奥运会于当地时间8月11日结束，历时16天，共设有329个奖牌项目。
   - **金牌榜前十名国家**：美国以126枚奖牌（40金、44银、42铜）位居榜首；中国紧随其后，获得91枚奖牌（40金、27银、24铜）；日本、澳大利亚、法国分别位列第三至第五名。
   - **中国代表团总体表现**：中国代表团在本次奥运会上表现出色，获得40枚金牌，总奖牌数达到91枚，位居第二。

## 2. 中国代表团详细表现：
   - **首金获得者及项目**：射击混合团体10米气步枪冠军，由来自江苏的盛李豪和来自浙江的黄雨婷夺得。
   - **前十枚金牌项目及获得者**：包括跳水女子双人三米板（昌雅妮、陈艺文）、射击男子10米气手枪（谢瑜）、跳水男子双人10米跳台（练俊杰、杨昊）、射击男子10米气步枪（盛李豪）、乒乓球混合双打（孙颖莎、王楚钦）、跳水女子双人10米跳台（全红婵、陈芋汐）、女子自由式小轮车公园赛（邓雅文）、男子100米自由泳（潘展乐）、男子50米步枪三姿（刘宇坤）。
   - **其他重要金牌项目及获得者**：包括田径女子20公里竞走（杨家玉）、男子双人三米板（王宗源、龙道一）、羽毛球混双（郑思维、黄雅琼）、乒乓球女单（陈梦）、羽毛球女子双打（陈清晨、贾一凡）、网球女子单打（郑钦文）、乒乓球男子单打（樊振东）、男子吊环（刘洋、邹敬园）、男子4×100米混合泳接力（徐嘉余、覃海洋、孙佳俊、潘展乐）、射击男子25米手枪速射（李越宏）、竞技体操男子双杠（邹敬园）、跳水女子10米台（全红婵、陈芋汐）、举重男子61公斤级（李发彬）、花样游泳集体技巧自选（中国队）、举重女子49公斤级（侯志慧）、皮划艇静水男子500米双人划艇（刘浩、季博文）、跳水男子3米板（谢思埸、王宗源）。

## 3. **新兴项目与突破**：
   - **新兴项目冠军**：如女子自由式小轮车公园赛冠军邓雅文。
   - **历史性突破**：男子100米自由泳冠军潘展乐打破世界纪录，实现该项目中国人零的突破。
'''

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
## markdown文档分块，每个块对应一个Document对象
def split_markdown(markdown_document: str, chunk_overlap = 30, chunk_size: int = 256) -> list[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(md_header_splits)
    return splits
docs = split_markdown(text)

print('docs len: %d\n' % len(docs))

#elastic_vector_search.add_documents(docs)

# Use the vectorstore as a retriever
score_threshold = 0.6
retriever = elastic_vector_search.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度得分阈值检索
    search_kwargs={"score_threshold": score_threshold, "k":3} # 设置相似度阈值和返回的最大文档数
)

# Retrieve the most similar text
retrieved_documents = retriever.invoke("新兴项目与突破")

# show the retrieved document's content
print(retrieved_documents[0].page_content)


from langchain_community.chat_models import ChatHunyuan
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI
from langchain_core.runnables import RunnableMap, RunnablePassthrough

## 1. 定义promot模板
ANSWER_TEMPLATE = """你是问答小助手，基于给出的[上下文]和[问题]，提供了一份最佳的回答，
回答要求是你经过深思熟虑、符合逻辑。“

[上下文]'''
{context}
'''

[问题]'''
{question}
'''
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

## 2. 定义文档组合模版
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


## 3. 初始化混元ChatHunyuan
tcloud_appid = config.get_settings().tcloud_appid
model = ChatHunyuan(model="hunyuan-pro", 
    hunyuan_app_id=tcloud_appid,
    hunyuan_secret_id=secretid,
    hunyuan_secret_key=secretkey,
    streaming=True)

rag_chain = (
    {"context": retriever | _combine_documents, "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | model
    | StrOutputParser()
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
add_routes(app, rag_chain, enable_feedback_endpoint=True, path="/hunyuan1")

rag_chain1 = (
    ANSWER_PROMPT
    | model
    | StrOutputParser()
)
add_routes(app, rag_chain1, enable_feedback_endpoint=True, path="/hunyuan2")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)