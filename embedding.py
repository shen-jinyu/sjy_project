from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# OllamaEmbedding

ollama_embed = OllamaEmbeddingFunction(             # 表明embeddings由本地Ollama服务生成,赋值给变量
            model_name="qwen3-embedding:0.6b",      # 指定嵌入模型,向量化模型
            url="http://127.0.0.1:11434")