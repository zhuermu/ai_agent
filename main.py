#!/usr/bin/env python
from operator import itemgetter
from typing import List, Tuple
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_community.chat_models import ChatHunyuan
from langchain_openai import ChatOpenAI
import config,os
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from local_langchain.hunyuan import HunyuanEmbeddings
from pydantic import BaseModel, Field
from langchain_elasticsearch import ElasticsearchStore

tcloud_secret_id = config.get_settings().tcloud_secret_id
tcloud_secret_key = config.get_settings().tcloud_secret_key
tcloud_appid = config.get_settings().tcloud_appid
ES_URL = config.get_settings().es_url
ES_API_KEY = config.get_settings().es_api_key

_TEMPLATE = """给定以下【对话历史]和一个[后续问题]，将后续问题重述为一个[独立的问题]，使用原始语言。
[对话历史]'''
{chat_history}
'''
[后续问题]'''
{question}
'''
独立问题:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """你是腾讯云解决方案小助手，你非常理解腾讯云的各种云产品的优势、技术原理和行业知识，基于以下[上下文]和[问题]，
为你提供了一份最佳的回答，回答要求是你经过深思熟虑、符合业务逻辑和全面详细的。如果[上下文]的内容不足以回答问题，请引导用户更换腾讯云相关的问题“

[上下文]'''
{context}
'''

[问题]'''
{question}
'''
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

embeddings = HunyuanEmbeddings(
    hunyuan_secret_id=tcloud_secret_id,
    hunyuan_secret_key=tcloud_secret_key,
    embedding_ctx_length=1024,
    region="ap-guangzhou",
)
vectorstore = ElasticsearchStore(
    embedding=embeddings,
    index_name="index_1024",
    es_url=ES_URL,
    es_api_key=ES_API_KEY,
)
# 设置相似度检索
score_threshold = 0.6    
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度得分阈值检索
    search_kwargs={"score_threshold": score_threshold, "k": 10}  # 设置得分阈值
)

model = ChatHunyuan(model="hunyuan-standard", 
    hunyuan_app_id=tcloud_appid,
    hunyuan_secret_id=tcloud_secret_id,
    hunyuan_secret_key=tcloud_secret_key,
    streaming=True)


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str



conversational_qa_rag_chain = (
    _inputs | _context | ANSWER_PROMPT | model | StrOutputParser()
)
chain = conversational_qa_rag_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(app, chain, enable_feedback_endpoint=True, path="/hunyuan")


ANSWER_TEMPLATE1 = """你是一名腾讯云工程师，目前需要协助客户对进行资源迁移，请根据腾讯云的cvm的产品型号和规则[腾讯云产品型号]和友商云机器[友商云产品型号]如下：

[腾讯云产品型号]'''
| 实例类型 | 子类型 | vCPU | 内存（GB） | 网络收发包（pps）（出+入） | 队列数 | 内网带宽能力（Gbps）（出+入） | 主频 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 标准型实例族 | 标准型 S8 | 2 | 8 | 40万 | 25万 | 2 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 4 | 8 | 80万 | 25万 | 4 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 4 | 16 | 80万 | 25万 | 4 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 8 | 16 | 160万 | 50万 | 8 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 8 | 32 | 160万 | 50万 | 8 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 16 | 32 | 320万 | 110万 | 16 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 16 | 64 | 320万 | 110万 | 16 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 32 | 64 | 640万 | 220万 | 32 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 32 | 128 | 640万 | 220万 | 32 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 56 | 256 | 1120万 | 400万 | 48 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 S8 | 56 | 512 | 1120万 | 400万 | 48 | 1.5GHz / 3.0GHz |
| 标准型实例族 | 标准型 SA5 | 2 | 2 | 25万 | 25万 | 2 | 1.5GHz / 3.1GHz |
| 标准型实例族 | 标准型 SA5 | 2 | 4 | 25万 | 25万 | 2 | 1.5GHz / 3.1GHz |
| 标准型实例族 | 标准型 SA5 | 4 | 8 | 30万 | 25万 | 4 | 1.5GHz / 3.1GHz |
| 标准型实例族 | 标准型 SA5 | 4 | 16 | 30万 | 25万 | 4 | 1.5GHz / 3.1GHz |
| 标准型实例族 | 标准型 SA5 | 8 | 16 | 30万 | 25万 | 4 | 1.5GHz / 3.1GHz |
| 标准型实例族 | 标准型 SA5 | 8 | 32 | 70万 | 25万 | 8 | 3.0GHz / 3.5GHz |
| 标准型实例族 | 标准型 SA5 | 8 | 64 | 70万 | 25万 | 8 | 3.0GHz / 3.5GHz |
| 标准型实例族 | 标准型 SA5 | 16 | 64 | 140万 | 50万 | 16 | 5.0GHz / 5.5GHz |
| 标准型实例族 | 标准型 SA5 | 16 | 128 | 140万 | 50万 | 16 | 5.0GHz / 5.5GHz |
| 标准型实例族 | 标准型 SA5 | 32 | 64 | 140万 | 50万 | 16 | 5.0GHz / 5.5GHz |
| 标准型实例族 | 标准型 SA5 | 32 | 128 | 140万 | 50万 | 16 | 5.0GHz / 5.5GHz |
| 标准型实例族 | 标准型 SA5 | 48 | 96 | 280万 | 100万 | 32 | 10.0GHz / 11.0GHz |
| 标准型实例族 | 标准型 SA5 | 48 | 192 | 280万 | 100万 | 32 | 10.0GHz / 11.0GHz |
| 标准型实例族 | 标准型 SA5 | 64 | 1152 | 2250万 | 800万 | 48 | 160.0GHz / 176.0GHz |
| 标准型实例族 | 标准型 SA5 | 128 | 2304 | 4500万 | 1600万 | 48 | 160.0GHz / 176.0GHz |
| 标准型实例族 | 标准型 SA4 | 8 | 16 | 90万 | 30万 | 8 | 2.5GHz / 3.7GHz |
| 标准型实例族 | 标准型 SA4 | 8 | 32 | 90万 | 30万 | 8 | 2.5GHz / 3.7GHz |
| 标准型实例族 | 标准型 SA4 | 16 | 32 | 180万 | 60万 | 16 | 4.0GHz / 4.8GHz |
| 标准型实例族 | 标准型 SA4 | 16 | 64 | 180万 | 60万 | 16 | 4.0GHz / 4.8GHz |
| 标准型实例族 | 标准型 SA4 | 32 | 64 | 370万 | 130万 | 32 | 8.0GHz / 9.6GHz |
| 标准型实例族 | 标准型 SA4 | 32 | 128 | 370万 | 130万 | 32 | 8.0GHz / 9.6GHz |
| 标准型实例族 | 标准型 SA4 | 48 | 192 | 2250万 | 400万 | 48 | 10.0GHz / 12.0GHz |
| 标准型实例族 | 标准型 SA4 | 48 | 384 | 2250万 | 400万 | 48 | 10.0GHz / 12.0GHz |
'''
[友商云产品型号]
'''
{question}
'''

请对表格内的机型规格推荐逐行匹配腾讯云的机型规格，推荐满足cpu核数及内存不得小于友商云机型。如果机型规格匹配的情况下有优惠的主力机型，优选优惠主力机型
推荐的腾讯云机型请附带推荐逻辑。输出格式为表格形式，表格包含列与输出表格相同，只输出表格内容，不需要输出其他信息。

[输出格式]
'''
| 友商云机型 | 腾讯云推荐机型｜ 推荐逻辑 |
| --- | --- | --- |

...


"""
ANSWER_PROMPT1 = ChatPromptTemplate.from_template(ANSWER_TEMPLATE1)

conversational_qa_chain = (
    ANSWER_PROMPT1 | model | StrOutputParser()
)

add_routes(app, conversational_qa_chain, enable_feedback_endpoint=True, path="/hunyuan1")


prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

os.environ["OPENAI_API_KEY"] = config.get_settings().anthropic_api_key

model_claude = ChatOpenAI(model='gpt-4o-2024-08-06', base_url='https://api.gptsapi.net/v1')

add_routes(
    app,
    prompt | model_claude,
    path="/openai",
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)