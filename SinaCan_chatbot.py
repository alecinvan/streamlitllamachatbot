import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="中加智能AI客服聊天助理")

# Replicate Credentials
with st.sidebar:
    st.title('中加智能AI客服聊天助理')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('请输入指定的密码:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('请输入您的凭证!', icon='⚠️')
        else:
            st.success('请继续输入您的聊天消息!', icon='👉')

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader('模型和参数')
    selected_model = st.sidebar.selectbox('请选择不同大语言模型（7B, 13B, 70B), 模型越大越精确，但是可能会稍慢：', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    
    temperature = st.sidebar.slider('温度值：较低（接近0）会导致生成的文本更加确定和一致，而较高会使生成的文本更加随机', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p：控制生成文本多样性的参数，它限制了生成文本中的词汇选择。较高值会使生成的文本更加多样，而较低会限制词汇的多样性。', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('最大长度：控制生成文本长度的参数，它定义了生成文本的最大长度', min_value=64, max_value=4096, value=512, step=8)
    
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "您好，请问有什么可以帮助您的，有各种问题都可以提问！"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "您好，请问有什么可以帮助您的，有各种问题都可以提问！"}]
st.sidebar.button('清除聊天记录', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "你是一个能干的助理。 你不以“用户”身份回复或冒充“用户”。 你仅以“助理”身份回复一次."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("请稍后，正在思考答案..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
