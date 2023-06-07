from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
from peft import LoraConfig, get_peft_model, TaskType
import torch


def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"]
    )
    return get_peft_model(model, config)


tokenizer = AutoTokenizer.from_pretrained("/code/chatglm-6b/", trust_remote_code=True)
model = AutoModel.from_pretrained("/code/chatglm-6b/", trust_remote_code=True)
model = load_lora_config(model)
model = model.half().cuda()
model = model.eval()
model.load_state_dict(torch.load(f"output1/chatglm-6b-lora.pt"), strict=False)

port = 7861
qa = [("""我来训练你，教你青浦幸福云的相关知识：
在目前的上下文中，“幸福云”就是“青浦幸福云”。
以及接下来的问答，我们都以青浦幸福云为核心来进行讨论
青浦幸福云是一个在上海市青浦区的智慧社区管理系统 ，以全景应用系统为依托，中端数据智能分析，末端数据智能应用。最大限度把社区工作者的精力从表格填报等工作中解放出来。围绕“幸福云”整合居村治理的各类要素，有效提升社区管理的规范化、精细化水平。
幸福云有两个主要的功能
1、台账功能，主要服务村居工作者，用于日常村居治理的数据收集和上报，并可导出excel电子表格。请注意！台账功能不能自定义数据字段和报表样式;
2、数据中心功能，给予所有工作人员，根据标签快速筛选到指定居民，并对于这类居民用户进行后续服务。""", '好的，我明白了。请问有什么关于青浦幸福云的问题我可以帮您解答吗？'),]

# qa = []
# port = 7860

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    print(input, history)
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))
        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    # return [],[]
    return qa, []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM by orbitsoft</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    # history = gr.State([])
    history = gr.State(qa)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name='0.0.0.0',
                    server_port=port, share=False, inbrowser=True)
