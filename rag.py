"""从向量数据库中检索相关知识点,将问题和参考资料拼接后送入大模型,然后返回答案"""

import chromadb
from tool.aaa_多模型env import llm_chat_with_history,add_message

# 本地化chroma客户端
client = chromadb.PersistentClient(path="./chroma_data")        # 连接本地持久化向量数据库,数据存储在path目录下

# collection名称
collection_name = "python1"                         # 指定数据库名称

# 访问指定collection
collection = client.get_collection(collection_name) # 获取指定数据库的数据

messages = []       # 创建空列表,用于存储完整的对话上下文

# system messages       # 通过调用函数,添加系统提示词
sys_msg = add_message("你是一个名叫Molly的教育专家，对于用户提出的问题，你需要按照给出的【参考资料】对问题进行回答。你的回答需要按照以下两个步骤：1.分析用户问题和参考资料，判断是否有【参考资料】可以解答用户的问题，如果有则说明【参考资料】的名称，如果没有，则首先告知用户没有任何可参考的资料，需要注意答案的准确性。2.根据资料内容对问题进行解答。", "system")
messages.append(sys_msg)    

# user message
query = '解释一下类和对象'  # 用户问题

result = collection.query(query_texts=[query])  # 将用户的问题送入向量模型,从向量数据库中检索关联记录,向量空间中查找最相似的N条
documents = result["documents"]     # 关联问题
answers = result["metadatas"]       # 关联问题的解答

"""result结构
{
"documents": [["问题1", "问题2", ...]],
"metadatas": [[{"answer": "答1"}, {"answer": "答2"}, ...]]
}"""

index_docs = []
for i in range(len(documents)):         # 遍历所有检索到的条目
    doc = documents[0][i]               # 问题描述
    answer = answers[0][i]["answer"]    # 解答
    index_doc = doc + "#" + answer      # 拼接
    index_docs.append(index_doc)        # 添加到空列表里


# 用户消息：输入问题+参考资料+检索出文档
message = "[用户问题]:\n" + query + \
        "\n [参考资料]:\n" + ("\n" + "="*50 + "\n\n").join(index_docs)

user_msg = add_message(message,"user")  # 将拼接完整的消息作为用户输入传给函数
messages.append(user_msg)               # 添加到消息列表里
# llm调用
_,result = llm_chat_with_history(       # 调用模型(返回思考内容:省略,返回输出值:需要)
    messages = messages,                # 消息对话传给参数
    model_name="glm-4-flash",           # 指定模型
    model_provider="zhipu",             # 指定供应商名
)

# 结果
print(result)               # 输出结果
