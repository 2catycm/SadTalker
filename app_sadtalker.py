import os, sys
import gradio as gr
from src.gradio_demo import SadTalker


class SadTalkerGUI:
    def __init__(
        self, checkpoint_path="checkpoints", config_path="src/config", warpfn=None
    ):
        sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
        gr.Markdown(
            "<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>"
        )

        # with gr.Row().style(equal_height=False):
        with gr.Row(equal_height=False):  # gradio 4.10.0
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem("Upload image"):
                        with gr.Row():
                            source_image = gr.Image(
                                label="Source image",
                                # source="upload",
                                type="filepath",
                                # elem_id="img2img_image").style(width=512)
                                elem_id="img2img_image",
                                width=512,
                            )

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem("Upload OR TTS"):
                        with gr.Column(variant="panel"):
                            driven_audio = gr.Audio(
                                label="Input audio",
                                # source="upload",
                                type="filepath",
                            )
                        
                        from tts_ui import make_tts_ui
                        with gr.Column(variant="panel"):
                            self.input_text_gr = make_tts_ui(driven_audio)

            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem("Settings"):
                        gr.Markdown(
                            "need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials"
                        )
                        with gr.Column(variant="panel"):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            pose_style = gr.Slider(
                                minimum=0,
                                maximum=46,
                                step=1,
                                label="Pose style",
                                value=0,
                            )  #
                            size_of_image = gr.Radio(
                                [256, 512],
                                value=256,
                                label="face model resolution",
                                info="use 256/512 model?",
                            )  #
                            preprocess_type = gr.Radio(
                                ["crop", "resize", "full", "extcrop", "extfull"],
                                value="crop",
                                label="preprocess",
                                info="How to handle input image?",
                            )
                            is_still_mode = gr.Checkbox(
                                label="Still Mode (fewer head motion, works with preprocess `full`)"
                            )
                            batch_size = gr.Slider(
                                label="batch size in generation",
                                step=1,
                                maximum=10,
                                value=2,
                            )
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            submit = gr.Button(
                                "Generate",
                                elem_id="sadtalker_generate",
                                variant="primary",
                            )

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(
                        label="Generated video",
                        #  format="mp4",
                        width=256,
                    )

        if warpfn:
            submit.click(
                fn=warpfn(sad_talker.test),
                inputs=[
                    source_image,
                    driven_audio,
                    preprocess_type,
                    is_still_mode,
                    enhancer,
                    batch_size,
                    size_of_image,
                    pose_style,
                ],
                outputs=[gen_video],
            )
        else:
            submit.click(
                fn=sad_talker.test,
                inputs=[
                    source_image,
                    driven_audio,
                    preprocess_type,
                    is_still_mode,
                    enhancer,
                    batch_size,
                    size_of_image,
                    pose_style,
                ],
                outputs=[gen_video],
            )


import socket
import socket
import threading
# Create a mutex lock
port_lock = threading.Lock()

def get_free_port():
    with port_lock:
        sock = socket.socket()
        sock.bind(('', 0))
        ip, port = sock.getsockname()
        sock.close()
        return port

demo = None
# last_response = None
if __name__ == "__main__":
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("<div align='center'> <h1> Talking Head WebUI </h1> </div>")
        # copy_button = gr.Button("Copy last response to text prompt")
        
        with gr.Tab("🏆 Video Content Conception Drafting 🚀"):
            # gr.Markdown("TODO, a chat interface")
            
            GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            # chat = model.start_chat(history=[])
            def llm_for_gr(message:str, history:list[str, str]):
                # response = chat.send_message(message, stream=True)
                # print(history)
                # history.append([message, None])
                history_for_google_model = []
                for user_text, model_text in history:
                    history_for_google_model.append(
                        dict(
                            role="user",
                            parts=[dict(text=user_text)]
                        ))
                    if model_text:
                        history_for_google_model.append(
                            dict(
                                role="model",
                                parts=[dict(text=model_text)]
                            ))
                history_for_google_model.append(
                    dict(
                            role="user",
                            parts=[dict(text=message)]
                        )
                )
                # print(history_for_google_model)
                response = model.generate_content(history_for_google_model, 
                                                  stream=True)
                full_response = ""
                for chunk in response:
                    # print(chunk.text)
                    # print("_"*80)
                    full_response+=chunk.text
                    yield full_response
                # nonlocal last_response
                # last_response = full_response
                
            chat_interface = gr.ChatInterface(fn=llm_for_gr, 
                                              examples=["Hello!", "Who are you? ", "Are you OK? "], 
                                              title="Gemini").queue()
            
        with gr.Tab("🎞️ Automate a talking head video"):
            sad_talker = SadTalkerGUI()
    # demo.queue(
    #     # concurrency_count=5, 
    #         #    max_size=20
    #            )
        # def handle_copy(*args, **kwargs):
        #     print(args)
        #     print(kwargs)
        #     return ""
        # btn = gr.Button("copy")
        # btn.click(handle_copy, inputs=[chat_interface], 
        #                 outputs=[sad_talker.input_text_gr])
    demo.launch(server_port=get_free_port(), 
                height=1000,
                # share=True
                share=False
                )
    # @demo.app.get("/hello")
    # def another_fastapi_not_gradio():
    #     return {"Hello": "World"}
