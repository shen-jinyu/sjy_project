"""通过导入的json文件,将文件构建成一个知识库,然后导入向量数据库"""

import chromadb
from embedding import ollama_embed
import json
from uuid import uuid4

# 本地化chroma客户端
client = chromadb.PersistentClient(path="./chroma_data")        # 创建一个持久化客户端,指定向量数据库的存储路径,存储到client（EphemeralClient仅内存临时存储）

def json_parse(file_path):                                      # 定义函数,解析用户提供的json文件,提取‘问题’和‘答案’列表
    with open(file_path, 'r', encoding='utf-8') as file:        # 打开这个文件,只读的方式读取这个文件，然后赋值给file
        data = json.load(file)                                  # 将file解析为python对象(列表或字典)，然后赋值给data

    keyword_list = []                   # 存储所有‘问题’的列表
    answer_list = []                    # 存储所有‘答案’的列表
    
    for item in data:                                           # 遍历data赋值给item
        k_qa_content = item['k_qa_content']                     # 从item中取出键为‘k_qa_content’的值，然后赋值给k_qa_content
        keyword, answer = k_qa_content.split('#', maxsplit=1)   # 分割值,按照第一个“#”分隔,maxsplit决定第几个分隔符分割,分别赋值给变量
        keyword_list.append(keyword)                            # 将变量添加到空问题列表
        answer_list.append(answer)                              # 将变量添加到空答案列表
    return keyword_list,answer_list                             # 返回两个列表

def check_collection_exists(collection_name):                   # 定义函数,检查指定名称的数据库是否存在
    """检查指定名称是否存在"""
    collections = client.list_collections()                     # 调用Chroma API获取当前数据库中的所有集合的信息，返回collections列表
    for elem in collections:                                    # 遍历collections列表
        if elem.name == collection_name:                        # 判断是否同名
            return True                                         # 同名返回True
    return False                                                # 否则循环结束返回False

def create_and_fill_collection(collection_name,file_path):      # 定义函数,根据json文件创建新的数据库和存储路径 
    """通过制定Collection名称和需要导入json文件构建Collection"""
    
    # 创建collection
    collection=client.create_collection(                        # 调用内置函数创建新集合
        name=collection_name,                                   # 集合名称,赋值给name
        embedding_function=ollama_embed                         # 表明embeddings由本地Ollama服务生成,赋值给变量
        )

    # 导入json
    keywords,answers = json_parse(file_path)                    # 调用定义的函数,得到问题和答案列表，分别赋值给变量
    # 验证
    assert len(keywords) == len(answers), "问题和回答长度不一致"  # 验证,保证一个问题对应一个答案


    # 添加文档
    collection.add(                                             # 将新的问题和答案添加到新的colltion中
        ids=[str(uuid4()) for _ in range(len(keywords))],       # 为每条文档生成唯一的id
        documents=keywords,                                     # 待检索的文本,这些问题会被向量化(与用户的问题进行比对)
        metadatas=[{"answer":ans} for ans in answers],          # 遍历答案列表,返回列表嵌套字典,附加信息,不会被向量化,检索时可一并返回
    )

    # print(collection.peek())      # 可查看collection前几条数据,用于调试

def get_collections_name():
    """查询chroma返回所有的collection名称列表"""
    collections = client.list_collections()
    return [col.name for col in collections]



if __name__ == '__main__':                                      # 判断是否直接运行本脚本
    collection_name="python2"                                   # 要创建的数据库名
    file_path="./datas/python2.json"                            # 原文件的文件路径

    if not check_collection_exists(collection_name):            # 判断要创建的数据库名是否存在
        create_and_fill_collection(collection_name,file_path)   # 不存在就调用上面的函数(创建新的数据库和文件路径)
        print(f"{collection_name} 创建成功！")                  # 用于提示数据库创建成功
    else:
        print(f"{collection_name} 已存在,创建失败！")           # 用于提示数据创建失败