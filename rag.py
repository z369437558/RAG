from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os

# 导入文本
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加载 Ollama 模型
llm = Ollama(model="llama2")

# 创建嵌入和检索器
embeddings = OllamaEmbeddings() 
faiss_index_path = "faiss_index"

if os.path.exists(faiss_index_path):
    # 从磁盘加载向量
    print("Loading vectors from disk...")
    vector = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    # 生成向量并保存到磁盘
    print("Generating and saving vectors...")
    vector = FAISS.from_documents(split_documents, embeddings)
    vector.save_local(faiss_index_path)

# 创建检索链
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
 
<context>
{context}
</context>
 
Question: {input}""")
 
document_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain
 
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 进行问答
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
