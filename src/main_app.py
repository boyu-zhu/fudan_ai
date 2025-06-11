import gradio as gr
from interface.model.kmeans_interface import kmeans_ui
from interface.model.svm_interface import svm_ui
from interface.model.cnn_interface import cnn_ui
from interface.model.mlp_interface import mlp_ui
from interface.model.mlp_interface import mlp_ui

from interface.interpret.shap_interface import shap_ui
from interface.interpret.lime_interface import lime_ui

# from
# from gradio_interfaces.model_b_interface import model_b_ui




def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 🔷 VisualFlow")
        gr.Markdown("欢迎使用 VisualFlow，一个用于可视化 AI 解释性的工具！")
        gr.HTML("""
        <style>
          #shap-block, #lime-block {
            background-color: white !important;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 0 8px rgba(0,0,0,0.05);
          }
        </style>
        """)
        with gr.Tabs():
            # Tab 1: 可视化页面
            with gr.Tab("可视化"):
                model_selector = gr.Dropdown(choices=["K-means", "SVM", "CNN", "MLP"], label="选择模型")

                kmeans_block = gr.Group(visible=True)  # K-means 模型的界面
                SVM_block = gr.Group(visible=False)
                CNN_block = gr.Group(visible=False)
                MLP_block = gr.Group(visible=False)



                # 绑定 Model A 的界面
                with kmeans_block:
                    kmeans_ui()


                # 绑定 Model B 的界面
                with SVM_block:
                    svm_ui()

                with CNN_block:
                    cnn_ui()

                with MLP_block:
                    mlp_ui()

                    # model_b_ui()
                    # pass

                # 控制哪个界面可见
                def toggle_model(model_name):
                    return (
                        gr.update(visible=(model_name == "K-means")),
                        gr.update(visible=(model_name == "SVM")),
                        gr.update(visible=(model_name == "CNN")),
                        gr.update(visible=(model_name == "MLP"))


                    )

                model_selector.change(
                    toggle_model,
                    inputs=model_selector,
                    outputs=[kmeans_block, SVM_block, CNN_block, MLP_block]
                )

            # Tab 2: 解释性页面
            with gr.Tab("解释性"):
                gr.Markdown("🔍 在这里你可以选择 LIME 或 SHAP 查看解释结果")

                explain_selector = gr.Dropdown(choices=["SHAP", "LIME"], label="选择解释方法")

                lime_block = gr.Group(visible=False)
                shap_block = gr.Group(visible=True, elem_id="shap-block")

                with shap_block:
                    gr.HTML("<div style='background-color: white; padding: 15px; border-radius: 10px;'>")
                    shap_ui()
                    gr.HTML("</div>")
                    
                with lime_block:
                    lime_ui()



                def toggle_explanation(name):
                    return (
                        gr.update(visible=(name == "SHAP")),
                        gr.update(visible=(name == "LIME")),

                    )

                explain_selector.change(
                    toggle_explanation,
                    inputs=explain_selector,
                    outputs=[shap_block, lime_block]
                )
    demo.launch()

if __name__ == "__main__":
    main()

# import gradio as gr

# with gr.Blocks(theme=gr.themes.Soft()) as demo:
#     with gr.Row():
#         with gr.Column(scale=0.2):
#             gr.Markdown("## 🔷 VisualFlow")  # 你也可以用图片替换这个 Markdown
#         with gr.Column(scale=0.8):
#             gr.Markdown("")  # 空占位，用于居中或布局
    
#     gr.Markdown("欢迎使用 VisualFlow，一个用于可视化 AI 解释性的工具！")

#     # 在这里添加你的主功能，比如上传图片，解释按钮等
#     with gr.Row():
#         image = gr.Image(type="pil")
#         output = gr.Textbox()

#     def analyze(img):
#         return "解释结果：图像内容非常复杂"

#     btn = gr.Button("解释图像")
#     btn.click(analyze, inputs=image, outputs=output)

# demo.launch()
