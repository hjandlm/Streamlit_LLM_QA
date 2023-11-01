'''
Description:  
Author: Huang J
Date: 2023-10-30 09:31:30
'''
import streamlit as st
import requests
import json
import os,sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration

torch.set_default_tensor_type(torch.cuda.HalfTensor)

# chat log
fs = open('chat_log.txt','a+')
fs.write(os.path.abspath(__file__))

# 外部知识获取
def get_reference(user_query,use_top_k=True,top_k=10,use_similar_score=True,threshold=0.7):
    
    # 外部知识检索接口
    SERVICE_ADD = ''
    ref_list = []
    user_query = user_query.strip()
    input_data = {}
    if use_top_k:
        input_data['query'] = user_query
        input_data['topk'] = top_k
        result = requests.post(SERVICE_ADD, json=input_data)
        res_json = json.loads(result.text)
        for i in range(len(res_json['answer'])):
            ref = res_json['answer'][i]
            ref_list.append(ref)
    elif use_similar_score:
        input_data['query'] = user_query
        input_data['topk'] = top_k
        result = requests.post(SERVICE_ADD, json=input_data)
        res_json = json.loads(result.text)
        for i in range(len(res_json['answer'])):
            maxscore = res_json['answer'][i]['prob']
            if maxscore > threshold: 
                ref = res_json['answer'][i]
                ref_list.append(ref)
    return ref_list

# clear 按钮
def on_btn_click():
    del st.session_state.messages

# 左边栏参数设置
def set_config():
    # 设置基本参数
    base_config = {"model_name":"","use_ref":"","use_topk":"","top_k":"","use_similar_score":"","max_similar_score":""}
    model_config = {'top_k':'','top_p':'','temperature':'','max_length':'','do_sample':""}
    
    with st.sidebar:
        model_name = st.radio(
            "模型选择：",
            ["baichuan2-13B-chat", "qwen-14B-chat","chatglm-6B","chatglm3-6B"],
            index=0
        )
        base_config['model_name'] = model_name
        set_ref = st.radio(
            "是否使用外部知识库：",
            ["是","否"],
            index=0,
        )
        base_config['use_ref'] = set_ref
        if set_ref=="是":
            set_topk_score = st.radio(
                '设置选择参考文献的方式：',
                ['use_topk','use_similar_score'],
                index=0,
                )
            if set_topk_score=='use_topk':
                set_topk = st.slider(
                    'Top_K', 1, 10, 5,step=1
                )
                base_config['top_k'] = set_topk
                base_config['use_topk'] = True
                base_config['use_similar_score'] = False
                set_score = st.empty()
            elif set_topk_score=='use_similar_score':
                set_score = st.slider(
                    "Max_Similar_Score",0.00,1.00,0.70,step=0.01
                )
                base_config['max_similar_score'] = set_score
                base_config['use_similar_score'] = True
                base_config['use_topk'] = False
                set_topk = st.empty()
            else:
                set_topk_score = st.empty()
                set_topk = st.empty()
                set_score = st.empty()
                
        sample = st.radio("Do Sample", ('True', 'False'))
        max_length = st.slider("Max Length", min_value=64, max_value=2048, value=1024)
        top_p = st.slider(
            'Top P', 0.0, 1.0, 0.7, step=0.01
        )
        temperature = st.slider(
            'Temperature', 0.0, 2.0, 0.05, step=0.01
        )
        st.button("Clear Chat History", on_click=on_btn_click)
        
    # 设置模型参数
    model_config['top_p']=top_p
    model_config['do_sample']=sample
    model_config['max_length']=max_length
    model_config['temperature']=temperature
    return base_config,model_config

# 设置模型输入格式
def set_input_format(model_name):

    if model_name=="baichuan2-13B-chat" or model_name=='baichuan2-7B-chat':
        input_format = "<reserved_106>{{query}}<reserved_107>"
    elif model_name=="qwen-14B-chat":
        input_format = """
        <|im_start|>system 
        你是一个乐于助人的助手。<|im_end|>
        <|im_start|>user
        {{query}}<|im_end|>
        <|im_start|>assistant"""
    elif model_name=="chatglm-6B":
        input_format = """{{query}}"""
    elif model_name=="chatglm3-6B":
        input_format = """
        <|system|>
        You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
        <|user|>
        {{query}}
        <|assistant|>\n
        """
    return input_format

# 加载模型和分词器
@st.cache_resource
def load_model(model_name):

    if model_name=="baichuan2-13B-chat":
        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",trust_remote_code=True)
        lora_path = ""
        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",trust_remote_code=True)
        model.to("cuda:0")
    elif model_name=="qwen-14B-chat":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat",trust_remote_code=True)
        lora_path = ""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat",trust_remote_code=True)
        model.to("cuda:1")
    elif model_name=="chatglm-6B":
        model = ChatGLMForConditionalGeneration.from_pretrained('THUDM/chatglm-6b',trust_remote_code=True)
        lora_path = ""
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',trust_remote_code=True)
        model.to("cuda:2")
    elif model_name=="chatglm3-6B":
        model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b',trust_remote_code=True)
        lora_path = ""
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b',trust_remote_code=True)
        model.to("cuda:3")
        
    # 加载lora包
    model = PeftModel.from_pretrained(model,lora_path)
    return model,tokenizer

# LLM chat
def llm_chat(model_name,model,tokenizer,model_config,query):
    response = ''
    top_k = model_config['top_k']
    top_p = model_config['top_p']
    max_length = model_config['max_length']
    do_sample = model_config['do_sample']
    temperature = model_config['temperature']
    
    if model_name=="baichuan2-13B-chat" or model_name=='baichuan-7B-chat':
        messages = []
        messages.append({"role": "user", "content": query})
        response = model.chat(tokenizer, messages)
    elif model_name=="qwen-14B-chat":
        response, history = model.chat(tokenizer, query, history=None, top_p=top_p, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature)
    elif model_name=="chatglm-6B":
        response, history = model.chat(tokenizer, query, history=None, top_p=top_p, max_length=max_length, do_sample=do_sample, temperature=temperature)
    elif model_name=="chatglm3-6B":
        response, history= model.chat(tokenizer, query, top_p=top_p, max_length=max_length, do_sample=do_sample, temperature=temperature)
    return response

if __name__=="__main__":
    
    #对话的图标
    user_avator = "🧑‍💻"
    robot_avator = "🤖"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    torch.cuda.empty_cache()
    base_config,model_config = set_config()
    model_name = base_config['model_name']
    use_ref = base_config['use_ref']
    model,tokenizer = load_model(model_name=model_name)
    input_format = set_input_format(model_name=model_name)

    st.header(f'Large Language Model：{model_name}')
    if use_ref=="是":
        col1, col2 = st.columns([5, 3])  
        with col1:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar=message.get("avatar")):
                    st.markdown(message["content"])
        if user_query := st.chat_input("请输入内容..."):
            with col1:  
                with st.chat_message("user", avatar=user_avator):
                    st.markdown(user_query)
                st.session_state.messages.append({"role": "user", "content": user_query, "avatar": user_avator})
                with st.chat_message("robot", avatar=robot_avator):
                    message_placeholder = st.empty()
                    use_top_k = base_config['use_topk']
                    if use_top_k:
                        top_k = base_config['top_k']
                        use_similar_score = base_config['use_similar_score']
                        ref_list = get_reference(user_query,use_top_k=use_top_k,top_k=top_k,use_similar_score=use_similar_score)
                    else:
                        use_top_k = base_config['use_topk']
                        use_similar_score = base_config['use_similar_score']
                        threshold = base_config['max_similar_score']
                        ref_list = get_reference(user_query,use_top_k=use_top_k,use_similar_score=use_similar_score,threshold=threshold)
                    if ref_list:
                        context = ""
                        for ref in ref_list:
                            context = context+ref['para']+"\n"
                        context = context.strip('\n')
                        query = f'''
                        上下文：
                        【
                        {context} 
                        】
                        只能根据提供的上下文信息，合理回答下面的问题，不允许编造内容，不允许回答无关内容。
                        问题：
                        【
                        {user_query}
                        】
                        '''
                    else:
                        query = user_query
                    query = input_format.replace("{{query}}",query)
                    max_len = model_config['max_length']
                    if len(query)>max_len:
                        cur_response = f'字数超过{max_len}，请调整max_length。'
                    else:
                        cur_response = llm_chat(model_name,model,tokenizer,model_config,query)
                    fs.write(f'输入：{query}')
                    fs.write('\n')
                    fs.write(f'输出：{cur_response}')
                    fs.write('\n')
                    sys.stdout.flush()
                    if len(query)<max_len:
                        if ref_list:
                            cur_response = f"""
                            大模型将根据外部知识库回答您的问题：<br>{cur_response}
                            """
                        else:
                            cur_response = f"""
                            大模型将根据预训练时的知识回答您的问题，存在编造事实的可能性。因此以下输出仅供参考：<br>{cur_response}
                            """
                    message_placeholder.markdown(cur_response)
                st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
            with col2:
                ref_list = get_reference(user_query)
                if ref_list:
                    for ref in ref_list:
                        ques = ref['ques']
                        answer = ref['para']
                        score = ref['prob']
                        question = f'{ques}--->score: {score}'
                        with st.expander(question):
                            st.write(answer)
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])
        if user_query := st.chat_input("请输入内容..."):
            with st.chat_message("user", avatar=user_avator):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query, "avatar": user_avator})
            with st.chat_message("robot", avatar=robot_avator):
                message_placeholder = st.empty()
                query = input_format.replace("{{query}}",user_query)
                max_len = model_config['max_length']
                if len(query)>max_len:
                    cur_response = f'字数超过{max_len}，请调整max_length。'
                else:
                    cur_response = llm_chat(model_name,model,tokenizer,model_config,query)
                fs.write(f'输入：{query}')
                fs.write('\n')
                fs.write(f'输出：{cur_response}')
                fs.write('\n')
                sys.stdout.flush()
                cur_response = f"""
                大模型将根据预训练时的知识回答您的问题，存在编造事实的可能性。因此以下输出仅供参考：{cur_response}
                """
                message_placeholder.markdown(cur_response)
                st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
                
                