import gradio as gr
from models.mlp import *


def mlp_ui():
    with gr.Tab("训练可视化"):
        with gr.Row():
            with gr.Column():
                dataset = gr.Dropdown(choices=["MNIST", "Circle"], 
                                    label="选择数据集", value="MNIST")
                layer_structure = gr.Textbox(label="网络层结构（逗号分隔，如128,64）", 
                                        value="10")
                epochs = gr.Slider(1, 200, value=5, step=1, 
                                label="训练轮数", interactive=True)
                lr = gr.Slider(0.001, 1.0, value=0.01, step=0.001,
                            label="学习率", interactive=True)
                btn = gr.Button("开始训练", variant="primary")
            
                sample_index_view = gr.Number(value=0, precision=0, label="查看样本编号")
                example_image = gr.Image(label="数据样本图像", height=280)
                view_btn = gr.Button("显示样本")
                view_btn.click(
                    fn=load_dataset_example,
                    inputs=[dataset, sample_index_view],
                    outputs=example_image
                )
            with gr.Column():
                out_loss = gr.Image(label="训练Loss 曲线", height=300)
            with gr.Column("激活可视化"):
                activation_image = gr.Image(label="逐层激活图", 
                                          height=400, interactive=False,type="filepath")
                with gr.Row():
                    gen_btn = gr.Button("生成激活帧")
                    gen_status = gr.Textbox(label="帧生成状态")
                    gen_btn.click(fn=generate_all_epoch_frames,
                            inputs=[layer_structure, dataset],
                            outputs=gen_status)
                    
                
                with gr.Row():
                    sample_index = gr.Number(value=0, precision=0, label="要播放的样本编号")
                    fps_slider = gr.Slider(1, 1000, value=10, step=1,
                                          label="播放速度 (FPS)", interactive=True)
                    play_button = gr.Button("播放", variant="secondary")
                    stop_button = gr.Button("停止", variant="stop")

                btn.click(fn=train_with_activation_visual,
                        inputs=[dataset, layer_structure, epochs, lr],
                        outputs=[out_loss])
                
                play_event =play_button.click(
                    fn=play_saved_frames_as_gif,
                    inputs=[dataset,sample_index, fps_slider],
                    outputs=activation_image
                )

                stop_button.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    cancels=[play_event]
                )
    with gr.Tab("过程可视化"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""MNIST样例输入""")
                    sketch_input = gr.ImageEditor(
                        label="数字绘画板",
                        image_mode="L",
                        brush=gr.Brush(colors=["#000000"], color_mode="fixed"),
                        height=280,
                        width=280,
                        type="pil",  
                    )
                    sketch_play = gr.Button("开始预测", variant="secondary")
                    gr.Markdown("""Circle样例输入""")
                    x1_input = gr.Slider(-1, 1, step=0.01, value=0.0, label="X 坐标")
                    x2_input = gr.Slider(-1, 1, step=0.01, value=0.0, label="Y 坐标")
                    circle_predict_btn = gr.Button("预测Circle结果")
                    
                    gr.Markdown("""中间神经元激活区域可视化""")
                    act_img = gr.Image(label="正激活区域图")
                    with gr.Row():
                        sel_layer_idx = gr.Number(value=0, precision=0, label="层编号 (从0开始)")
                        sel_neuron_idx = gr.Number(value=0, precision=0, label="神经元编号")
                        with  gr.Column():
                            act_btn = gr.Button("显示正激活区域")
                            show_all_btn = gr.Button("显示全部正激活区域", variant="primary")
                        act_btn.click(
                            fn=show_neuron_activation_region,
                            inputs=[layer_structure, sel_layer_idx, sel_neuron_idx],
                            outputs=act_img
                        )
                        show_all_btn.click(
                        fn=generate_all_neuron_activation_grid,
                        inputs=[layer_structure],
                        outputs=act_img
                    )
                        
                with gr.Column():
                    prediction_label = gr.Label(label="预测结果", num_top_classes=3)
                    sketch_output = gr.Image(label="激活可视化", height=300, type="filepath")  # 直接显示GIF
                    prediction_plot = gr.Plot(label="预测概率分布")
                    
                    
                    with gr.Row():
                        sketch_fps = gr.Slider(1, 20, value=10, step=1,
                                            label="播放速度 (FPS)", interactive=True)
                        sketch_stop = gr.Button("停止", variant="stop")
                        
                        event = sketch_play.click(
                            fn=process_sketch_input,
                            inputs=[sketch_fps,sketch_input, layer_structure],
                            outputs=[sketch_output, prediction_label, prediction_plot]
                        )

                        sketch_stop.click(
                            fn=None,
                            inputs=None,
                            outputs=None,
                            cancels=[event]
                        )
            circle_predict_btn.click(
                        fn=process_circle_input,
                        inputs=[x1_input, x2_input, layer_structure],
                        outputs=[sketch_output, prediction_label, prediction_plot]
                    )