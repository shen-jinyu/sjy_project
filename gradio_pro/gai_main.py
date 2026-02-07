import gradio as gr
from gai_rag import query_rag
from tool.aaa_多模型env import add_message
from file_to_chroma import check_collection_exists,create_and_fill_collection,get_collections_name
# 上面是导入文件内的函数，分别是检查数据库是否已经存在,创建并存储新的数据库和路径

# 事件相关或其他函数定义在这里

# 清空文本块(参数和返回值由 组件inputs和outputs决定的)
def clear_input():                                  # 定义函数,用于重置输入框
    return ""       # 代码给文本框重新赋值(填充)

def send_message(query,history):                        # 定义发送消息函数,模拟聊天机器人的消息处理流程,两个参数(用户输入,历史对话)

        
    if len(history) == 0:

        # system messages 通过调用函数,添加系统提示词
        sys_msg = add_message("你是一个名叫Molly的教育专家，对于用户提出的问题，你需要按照给出的【参考资料】对问题进行回答。你的回答需要按照以下两个步骤：1.分析用户问题和参考资料，判断是否有【参考资料】可以解答用户的问题，如果有则说明【参考资料】的名称，如果没有，则首先告知用户没有任何可参考的资料，需要注意答案的准确性。2.根据资料内容对问题进行解答。", "system")
        history.append(sys_msg)    

    history.append({"role":"user","content":query})     # 用户问题追加到历史对话
    return '',history
    
def stream_message(history,collection_name):     # 发送消息后级联触发函数

    # 获取最后一条用户消息
    query = history[-1]["content"][0]["text"]                #获取用户消息

    gen = query_rag(query,history,collection_name=collection_name)      # stream方式调用返回generator

    # history内容是chatbot显示对话(不包含检索文档)
    # query_rag中的messages在LLM对话应用内容(包含检索文档)
    
    # history添加空的系统回复
    history.append({"role":"assistant","content":""})     # 把AI回答追加到历史对话中
    thingking_out = False
    for think,token in gen:
        # stream遍历过程中不断在刚添加系统消息上进行文本拼接
        if think:
            thingking_out = True
            history[-1]["content"] += think
        if token:
            # 清空thingking输出
            if thingking_out:
                history[-1]['content'] = ""
                thingking_out = False
            history[-1]['content'] += token

        # yield返回history
        yield history                                      # 返回历史对话

# 添加collection实现函数
def create_colletion(collection_name,file_path):               # 定义函数,接收数据库和路径
    if not check_collection_exists(collection_name):           # 判断数据库名是否存在
        create_and_fill_collection(collection_name,file_path)  # 如果不存在就调用函数创建知识库
        gr.Info(f"{collection_name} 创建成功！")                # 显示通知创建成功
        return "",None                                         # 返回空值,用于清空输入框和文件上传组件
    else:
        gr.Warning(f"{collection_name} 已存在,创建失败！")   # 如果存在,就提示存在
        return collect_name,file_path                       # 返回数据库名和文件路径(用于更改)

# 填充列表
def fill_collections():
    names = get_collections_name()
    # gradio的update方法更新output组件中choices属性值
    return gr.update(choices=names,multiselect=False)


with gr.Blocks() as demo:                              # 创建一个自由布局的Gradio应用

    gr.Markdown("# 智能学习助教")                       # 添加标题

    with gr.Row():                                      # 创建水平分栏,内部组件将左右分布
        with gr.Column(scale=7):                               # 在Row内创建左侧垂直列

            chatbot = gr.Chatbot(label="对话框")        # 创建聊天显示区域,格式为列表嵌套字典    
            
            user_input = gr.Textbox(label="文本框")     # 文本聊条框,用于用户提问
            
            clear_msg = gr.Button("clear")              # 按钮为清空输入框内容
            
            gr.Examples(["解释一下类和对象","解释一下embedding模型"],user_input)    # 提供两个示例问题按钮,自动填充到user_input中

        with gr.Column(scale=3):                       # 创建右侧垂直列
            # 文件上传
            upload=gr.File(label="上传文件")            # 允许上传json文件,变量存储临时文件路径 
            # 指定Collection    
            collect_name=gr.Textbox(label="数据库名")   # 用户要创建的数据库名称
            # 构建数据库的按钮
            btn_create = gr.Button("开始构建")          # 构建按钮
            # 选择数据库    
            gr.Markdown("#### 选择数据库")              # 构建标题
            # 数据库选项
            collection_list = gr.Dropdown(label="数据库名",allow_custom_value=True)  # 构建数据库选项,下拉选择
            # 刷新按钮
            btn_refresh = gr.Button("刷新")                   # 构建刷新按钮

    # 事件注册处理代码写在Blocks()语句块里面
     
    # 发送消息处理
    user_input.submit(                      # 绑定回车提交
        fn=send_message,                    # 按下回车,触发send_message
        inputs=[user_input,chatbot],        # 输入框内的信息
        outputs=[user_input,chatbot],       # 输出信息
    ).then(
        fn=stream_message,                  # 按下回车,触发stream_message
        inputs=[chatbot,collection_list],                   # 输入框内的信息
        outputs=[chatbot],                  # 输出信息
    )

    # btn_create点击时间注册
    btn_create.click(                   # 点击开始构建
        fn=create_colletion,            # 调用函数构建数据库
        inputs=[collect_name,upload],   # 输入文本框的值和文件路径
        outputs=[collect_name,upload],  # 将返回值分别赋值给数据库名和路径(实现自动清空)
    )

    # 刷新按钮点击事件
    btn_refresh.click(
        fn = fill_collections,
        inputs = [],
        outputs = [collection_list]
    )

    # 事件处理方法
    clear_msg.click(fn=clear_input,                # 实现清空按钮的功能
                    inputs=[],                     # 事件处理函数没有输入参数
                    outputs=[user_input],          # 事件处理函数有返回值
    ) 
    

demo.launch()       # 启动Gradio服务


