#!/usr/bin/env python3
"""
ATLAS Multihop QA Benchmarking
基于atlas_multihopqa.ipynb改写的完整脚本版本
"""

import os
import sys
import torch
import gc
import traceback
from configparser import ConfigParser
from openai import OpenAI
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore import create_embeddings_and_index
from atlas_rag.evaluation import BenchMarkConfig
from atlas_rag import setup_logger
from atlas_rag.retriever import HippoRAG2Retriever
from atlas_rag.evaluation import RAGBenchmark
from atlas_rag.llm_generator import LLMGenerator

def setup_environment():
    """设置运行环境"""
    print("🔧 设置运行环境...")
    print(f"Python版本: {sys.version}")
    
    # 设置为CPU模式
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(f"使用设备: CPU")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 设置PyTorch线程数以避免内存问题
    torch.set_num_threads(2)
    print(f"PyTorch线程数: {torch.get_num_threads()}")

def setup_models():
    """设置编码器和LLM模型"""
    print("\n🤖 设置模型...")
    
    try:
        # 1. 设置编码器模型
        encoder_model_name = 'all-MiniLM-L6-v2'
        sentence_model = SentenceTransformer(encoder_model_name, device='cpu')
        sentence_encoder = SentenceEmbedding(sentence_model)
        print(f"编码器模型: {encoder_model_name}")
        print(f"模型设备: {sentence_model.device}")
        
        # 2. 设置LLM模型
        config = ConfigParser()
        config.read('config.ini')
        
        reader_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=config['settings']['DEEPINFRA_API_KEY']
        )
        llm_generator = LLMGenerator(client=client, model_name=reader_model_name)
        print(f"LLM模型: {reader_model_name}")
        print("API端点: DeepInfra")
        
        return sentence_encoder, encoder_model_name, llm_generator, reader_model_name
    
    except Exception as e:
        print(f"❌ 模型设置失败: {e}")
        raise

def load_data(sentence_encoder, encoder_model_name):
    """加载和处理数据"""
    print("\n📊 加载数据...")
    
    try:
        # 使用与提取时相同的设置，但减少batch size
        keyword = 'musique_sample'
        working_directory = './working_data'
        
        print(f"数据集: {keyword}")
        print(f"工作目录: {working_directory}")
        
        # 使用更小的batch size以减少内存使用
        data = create_embeddings_and_index(
            sentence_encoder=sentence_encoder,
            model_name=encoder_model_name,
            working_directory=working_directory,
            keyword=keyword,
            include_concept=True,
            include_events=True,
            normalize_embeddings=True,
            text_batch_size=8,  # 减少到8
            node_and_edge_batch_size=8,  # 减少到8
        )
        
        # 清理内存
        gc.collect()
        print("✅ 数据加载完成，内存已清理")
        
        return data, keyword
    
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("🔧 尝试使用更小的batch size...")
        traceback.print_exc()
        
        # 尝试使用更小的batch size
        try:
            data = create_embeddings_and_index(
                sentence_encoder=sentence_encoder,
                model_name=encoder_model_name,
                working_directory=working_directory,
                keyword=keyword,
                include_concept=True,
                include_events=True,
                normalize_embeddings=True,
                text_batch_size=4,  # 进一步减少
                node_and_edge_batch_size=4,
            )
            gc.collect()
            return data, keyword
        except Exception as e2:
            print(f"❌ 即使使用小batch size也失败: {e2}")
            raise

def setup_benchmark(keyword, reader_model_name, encoder_model_name):
    """配置benchmark测试"""
    print("\n⚙️ 配置benchmark...")
    
    try:
        benchmark_config = BenchMarkConfig(
            dataset_name=keyword,
            question_file="benchmark_data/musique.json",
            include_concept=True,
            include_events=True,
            reader_model_name=reader_model_name,
            encoder_model_name=encoder_model_name,
            number_of_samples=3,  # 减少样本数量
        )
        
        logger = setup_logger(benchmark_config)
        return benchmark_config, logger
    
    except Exception as e:
        print(f"❌ Benchmark配置失败: {e}")
        raise

def run_benchmark(data, llm_generator, sentence_encoder, benchmark_config, logger):
    """运行benchmark测试"""
    print("\n🚀 开始benchmark测试...")
    
    try:
        # 初始化检索器
        hipporag2_retriever = HippoRAG2Retriever(
            llm_generator=llm_generator,
            sentence_encoder=sentence_encoder,
            data=data,
            logger=logger
        )
        
        # 运行benchmark
        benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
        benchmark.run([hipporag2_retriever], llm_generator=llm_generator)
        
        # 清理内存
        gc.collect()
        
    except Exception as e:
        print(f"❌ Benchmark运行失败: {e}")
        traceback.print_exc()
        raise

def main():
    """主函数"""
    print("=" * 60)
    print("🔥 ATLAS Multihop QA Benchmarking")
    print("=" * 60)
    
    try:
        # 1. 设置环境
        setup_environment()
        
        # 2. 设置模型
        sentence_encoder, encoder_model_name, llm_generator, reader_model_name = setup_models()
        
        # 3. 加载数据
        data, keyword = load_data(sentence_encoder, encoder_model_name)
        
        # 4. 设置benchmark
        benchmark_config, logger = setup_benchmark(keyword, reader_model_name, encoder_model_name)
        
        # 5. 运行benchmark
        run_benchmark(data, llm_generator, sentence_encoder, benchmark_config, logger)
        
        print("\n✅ Benchmark测试完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了程序")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 