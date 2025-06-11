import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import gradio as gr
from interpret.shap_interpreter import shapley  # 假设shapley类在shap_interpreter.py中定义

warnings.filterwarnings("ignore")

global_data = {
    'dataset_name': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'target_names': None,
    'model': None,
    'model_C': 1.0,
    'model_penalty': 'l2',
}



def load_dataset(name):
    if name == 'Iris':
        data = load_iris(as_frame=True)
    elif name == 'Breast Cancer':
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError("Unsupported dataset")
    return data.data, data.target, data.target_names

def fit_model(dataset_name):
    X, y, target_names = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=2022
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=global_data['model_C'], penalty=global_data['model_penalty'], solver='liblinear')
    )
    model.fit(X_train, y_train)
    
    # 保存全局
    global_data.update({
        'dataset_name': dataset_name,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'target_names': target_names,
        'model': model,
    })
    return f"数据集 {dataset_name} 训练完成，样本数：训练集{len(X_train)}，测试集{len(X_test)}"

def explain(sample_idx, M_samples):
    # 检查是否已训练模型
    if global_data['model'] is None:
        return "请先选择数据集并点击 Fit 模型按钮进行训练。", pd.DataFrame()

    X_test = global_data['X_test']
    y_test = global_data['y_test']
    X_train = global_data['X_train']
    target_names = global_data['target_names']
    f = global_data['model']
    x = X_test.iloc[sample_idx].values
    feature_names = X_test.columns

    # Shapley 估计
    shapval = shapley(X_train, lambda x_: f.predict_proba(x_)[0])
    MC = shapval.estimate(x, M=int(M_samples))

    # 真实标签和预测
    true_label = target_names[y_test.iloc[sample_idx]]
    pred_probs = f.predict_proba(X_test.iloc[sample_idx].values.reshape(1, -1))[0]
    pred_label = target_names[np.argmax(pred_probs)]
    prob_str = ", ".join([f"{target_names[i]}: {pred_probs[i]:.2f}" for i in range(len(pred_probs))])

    # 绘制柱状图
    plt.figure(figsize=(8, 5))
    idx = np.arange(len(feature_names))
    plt.bar(idx, MC, width=0.6, label='Shapley Value Estimate')
    plt.xticks(idx, feature_names, rotation=90)
    plt.ylabel('Shapley Value')
    plt.title(f'Shapley Values for Sample {sample_idx}')
    plt.legend()
    plt.tight_layout()

    import io
    import base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 输出表格
    df = pd.DataFrame({
        'Feature': feature_names,
        'Shapley Value': MC
    })

    # 结果总结
    result_summary = f"""
    <h3>Sample {sample_idx} Explanation</h3>
    <ul>
        <li><b>True Label:</b> {true_label}</li>
        <li><b>Predicted Label:</b> {pred_label}</li>
        <li><b>Model Hyperparameters:</b> C={global_data['model_C']}, penalty={global_data['model_penalty']}</li>
    </ul>
    """

    # 横向布局，表格左，图右
    html_output = f"""
    <div style="display:flex; gap:20px;">
        <div style="flex: 1; overflow-x:auto;">
            {df.to_html(index=False)}
        </div>
        <div style="flex: 1;">
            <img src='data:image/png;base64,{image}'/>
        </div>
    </div>
    <hr/>
    {result_summary}
    """

    return html_output, df

def shap_ui():
    with gr.Row():
        gr.Markdown("## Shapley 值估计器")
        dataset_selector = gr.Dropdown(choices=['Iris', 'Breast Cancer'], value='Iris', label="选择数据集")
        fit_button = gr.Button("Fit 模型",  variant="primary")
        fit_output = gr.Textbox(label="训练状态", interactive=False)

    with gr.Row():
        sample_slider = gr.Slider(0, 0, step=1, label='Sample Index')  # max会动态更新
        M_samples_slider = gr.Slider(50, 1000, step=50, value=200, label='Monte Carlo Sample Size (M)')
        explain_button = gr.Button("解释预测",  variant="primary")

    output_html = gr.HTML(label="Shapley Value Visualization & Info")
    output_table = gr.Dataframe(label="Shapley Values Table")

    # 绑定事件
    def update_sample_slider(dataset_name):
        # 加载数据集获取测试集大小
        X, y, _ = load_dataset(dataset_name)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=2022)
        return gr.update(max=len(X_test)-1, value=0)

    dataset_selector.change(fn=update_sample_slider, inputs=dataset_selector, outputs=sample_slider)

    fit_button.click(fn=fit_model, inputs=dataset_selector, outputs=fit_output)

    explain_button.click(fn=explain, inputs=[sample_slider, M_samples_slider], outputs=[output_html, output_table])
