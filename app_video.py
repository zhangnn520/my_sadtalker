import os, sys
import gradio as gr
from src.gradio_demo import SadTalker  


try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

    

def sadtalker_demo(checkpoint_path='checkpoints', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> SadTalker-Video-Lip-Sync </span> </h2> \
<h3>  一键包制作： 进化洲(公众号、视频号：进化洲 ，wx: jinhuazhou2023 ) </span> </h3> \
<h4>  免责声明：本作品仅作为娱乐目的发布，可能造成的后果与作者、贡献者无关。 </span> </h4> \
                <a style='font-size:18px;color: #efefef' href='https://github.com/Zz-ww/SadTalker-Video-Lip-Sync'> Github    </div>")

        
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                with gr.Tabs(elem_id="sadtalker_source_vedio"):
                    with gr.TabItem('上传视频'):
                        with gr.Row():
                            source_vedio = gr.Video(label="视频", source="upload", type="filepath", elem_id="img2img_image").style(width=256)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('上传音频'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="音频", source="upload", type="filepath")

                        # if sys.platform != 'win32' and not in_webui:
                        #     from src.utils.text2speech import TTSTalker
                        #     tts_talker = TTSTalker()
                        #     with gr.Column(variant='panel'):
                        #         input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                        #         tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                        #         tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])
                        #
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('设置'):
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            preprocess_type = gr.Radio(['none', 'lip','face'], value='face', label='增强方式 enhancer（none: 无， lip:嘴唇， face:脸部）', info="How to enhancer?")
                            batch_size = gr.Slider(label="batch size", step=1, maximum=10, value=1)
                            submit = gr.Button('生成', elem_id="sadtalker_generate", variant='primary')
                            
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="结果视频", format="mp4").style(width=256)

        if warpfn:
            submit.click(
                        fn=warpfn(sad_talker.test), 
                        inputs=[source_vedio,
                                driven_audio,
                                preprocess_type,
                                batch_size
                                ], 
                        outputs=[gen_video]
                        )
        else:
            submit.click(
                        fn=sad_talker.test, 
                        inputs=[source_vedio,
                                driven_audio,
                                preprocess_type,
                                batch_size
                                ], 
                        outputs=[gen_video]
                        )

    return sadtalker_interface
 

if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.queue()
    demo.launch(share=True)


