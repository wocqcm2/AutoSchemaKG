import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 注释掉CUDA设置，使用CPU
from openai import OpenAI
from atlas_rag.llm_generator import LLMGenerator
from configparser import ConfigParser
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding, NvEmbed
from neo4j import GraphDatabase
import faiss
import datetime
import logging
from logging.handlers import RotatingFileHandler
from atlas_rag.retriever.lkg_retriever.lkgr import LargeKGRetriever
from atlas_rag.retriever.lkg_retriever.tog import LargeKGToGRetriever
from atlas_rag.kg_construction.neo4j.neo4j_api import LargeKGConfig, start_app

# 使用与FAISS索引相同的模型和设置（CPU模式）
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # 与索引创建时相同
sentence_encoder = SentenceEmbedding(sentence_model)

# 加载OpenAI API配置
config = ConfigParser()
config.read('config.ini')

# 使用OpenAI模型（gpt-4o）
retriever_model_name = "gpt-4o"
reader_model_name = "gpt-4o"

# 创建OpenAI客户端
client = OpenAI(
    api_key=config['settings']['OPENAI_API_KEY'],
)

# 创建LLM生成器
llm_generator = LLMGenerator(client=client, model_name=reader_model_name)
retriever_llm_generator = LLMGenerator(client=client, model_name=retriever_model_name)

# prepare necessary objects for instantiation of LargeKGRetriever: neo4j driver, faiss index etc.
neo4j_uri = "bolt://localhost:8012" # use bolt port for driver connection
user = "neo4j"
password = "admin2024"
keyword = 'CICGPC_Glazing_ver1.0a' # can be wiki or pes2o  # keyword to identify the cc_en dataset
driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

text_index = faiss.read_index(f"import/CICGPC_Glazing_ver1.0a/precompute/CICGPC_Glazing_ver1.0a_text_faiss.index", faiss.IO_FLAG_MMAP)
node_index = faiss.read_index(f"import/CICGPC_Glazing_ver1.0a/precompute/CICGPC_Glazing_ver1.0a_eventTrue_conceptTrue_all-MiniLM-L6-v2_node_faiss.index", faiss.IO_FLAG_MMAP)

# setup logger
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
log_file_path = f'./log/LargeKGRAG_CICGPC_Glazing_ver1.0a.log'
logger = logging.getLogger("LargeKGRAG")
logger.setLevel(logging.INFO)
max_bytes = 50 * 1024 * 1024  # 50 MB
if not os.path.exists(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
logger.addHandler(handler)

# 创建大规模KG检索器（使用OpenAI API和CPU模式）
retriever = LargeKGRetriever(
    keyword=keyword,
    neo4j_driver=driver,
    llm_generator=retriever_llm_generator,  # 使用OpenAI API
    sentence_encoder=sentence_encoder,      # CPU模式
    node_index=node_index,
    passage_index=text_index,
    topN=5,
    number_of_source_nodes_per_ner=10,
    sampling_area=250,
    logger=logger
)
# 创建ToG检索器（全部使用CPU和OpenAI API）
tog_retriever = LargeKGToGRetriever(
    keyword=keyword,
    neo4j_driver=driver,
    topN=5,
    Dmax=2,
    Wmax=3,
    llm_generator=retriever_llm_generator,  # 使用OpenAI API
    sentence_encoder=sentence_encoder,      # CPU模式
    filter_encoder=SentenceEmbedding(SentenceTransformer('all-MiniLM-L6-v2', device='cpu')),  # 与索引相同的模型
    node_index=node_index,
    logger=logger
)

# 配置大规模KG RAG系统（使用OpenAI API）
large_kg_config = LargeKGConfig(
    largekg_retriever=tog_retriever,        # 使用ToG检索器
    reader_llm_generator=llm_generator,     # 使用OpenAI API进行文本生成
    driver=driver,                          # Neo4j数据库连接
    logger=logger,                          # 日志记录器
    is_felm=False,                         # 不是FELM评估
    is_mmlu=False,                         # 不是MMLU评估
)

# 启动FastAPI服务器
print("🚀 启动ATLAS RAG API服务器...")
print("📊 使用模型: OpenAI gpt-4o")
print("💾 数据库: Neo4j (CICGPC Glazing数据集)")  
print("🌐 服务地址: http://0.0.0.0:10088")
start_app(user_config=large_kg_config, host="0.0.0.0", port=10088, reload=False)