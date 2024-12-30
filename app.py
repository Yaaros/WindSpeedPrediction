# app.py

import json
import math
import os
import random

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from data import select_available_features, handle_outliers, create_sequences
from models import (
    LSTMModelMultistep,
    AttentionLSTMModel,
    TemporalFusionTransformerModel,
)  # 确保导入 AttentionLSTMModelMultistep 类
from utils.plot_utils import cache_plot_predictions

# 创建 cache 目录（如果不存在）
CACHE_DIR = './cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


# 加载参数函数
def load_params(filepath='Params.json'):
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params


# 加载测试数据函数，适配 main.py 的数据处理流程
def load_test_data(meters, selected_features, SEQ_LENGTH, PRED_STEPS):
    test_data_path = f"./dataset/test-windspeed-of-{meters}m.csv"

    if not os.path.exists(test_data_path):
        return None, None, "测试数据文件不存在。"

    df = pd.read_csv(test_data_path)

    # 处理缺失值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # 处理异常值
    data = handle_outliers(df, ['SpeedAvg', 'DirectionAvg', 'TemperatureAvg', 'PressureAvg', 'HumidtyAvg'])

    # 特征工程：计算压力差、温度差和湿度差，并填补height
    data['PressureDelta'] = data['PressureMax'] - data['PressureAvg']
    data['TemperatureDelta'] = data['TemperatureMax'] - data['TemperatureAvg']
    data['HumidtyDelta'] = data['HumityMax'] - data['HumidtyAvg']
    data['height'] = meters

    # 特征选择
    available_features, missing_features = select_available_features(data, selected_features)
    if missing_features:
        return None, None, f"以下选择的特征在数据集中未找到: {missing_features}"

    # 确保不包含目标变量 'SpeedAvg'
    if 'SpeedAvg' in available_features:
        available_features.remove('SpeedAvg')
    if meters != 10:
        available_features.append('Speed Avg 10m')
    # 提取特征和目标
    features = data[available_features].values
    target = data['SpeedAvg'].values  # 确认 'SpeedAvg' 作为目标变量
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(features)

    # 窗口化处理，确保与训练时一致
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, target, SEQ_LENGTH, PRED_STEPS)

    return X_test_seq, y_test_seq, None


# 模型预测函数
def predict_and_evaluate(model_path, model_type, X_test, device, SEQ_LENGTH, PRED_STEPS):
    """
    根据模型类型加载模型并进行预测。

    参数：
    - model_path (str): 模型文件路径。
    - model_type (str): 模型类型（Linear, LSTM, AttentionLSTM, Transformer）。
    - X_test (numpy.ndarray): 测试特征数据。
    - device (torch.device): 设备（CPU 或 CUDA）。
    - SEQ_LENGTH (int): 输入序列长度。
    - PRED_STEPS (int): 预测步数。

    返回：
    - y_pred (numpy.ndarray): 预测结果。
    - error (str): 错误信息（如有）。
    """
    file_extension = os.path.splitext(model_path)[1]  # 获取文件扩展名
    if model_type == "Linear":
        if file_extension != ".pkl":
            return None, f"Linear 模型文件格式应为 .pkl，但找到了 {file_extension}"
        try:
            # 加载线性回归模型
            linear_model = joblib.load(model_path)
            y_pred = linear_model.predict(X_test.reshape(X_test.shape[0], -1))
            y_pred = y_pred.reshape(-1, 1) if PRED_STEPS == 1 else y_pred
            return y_pred, None
        except Exception as e:
            return None, f"加载线性回归模型失败: {str(e)}"

    elif model_type in ["LSTM", "AttentionLSTM", "Transformer"]:
        if file_extension != ".pth":
            return None, f"{model_type} 模型文件格式应为 .pth，但找到了 {file_extension}"
        try:
            if model_type == "LSTM":
                model = LSTMModelMultistep(
                    input_size=X_test.shape[2],  # 特征数量
                    hidden_size=128,
                    num_layers=2,
                    m_steps=PRED_STEPS
                )
            elif model_type == "AttentionLSTM":
                model = AttentionLSTMModel(
                    input_size=X_test.shape[2],
                    hidden_size=128,
                    num_layers=2,
                    m_steps=PRED_STEPS
                )
            elif model_type == "Transformer":
                model = TemporalFusionTransformerModel(
                    input_size=X_test.shape[2],  # 特征数量
                    d_model=32,
                    nhead=4,
                    num_encoder_layers=2,
                    num_decoder_layers=2,
                    dim_feedforward=256,
                    m_steps=PRED_STEPS
                )


            # 加载模型权重
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            with torch.no_grad():
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                # 确保输入为 [batch_size, seq_length, num_features]
                # 打印输入形状用于调试
                print(f"输入形状: {X_tensor.shape}")
                y_pred = model(X_tensor).cpu().numpy()

            return y_pred, None
        except Exception as e:
            return None, f"加载{model_type}模型失败: {str(e)}"
    else:
        return None, "不支持的模型类型。"


# 清理 cache 目录中的临时文件
def clear_cache():
    cache_dir = CACHE_DIR
    for f in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"无法删除文件 {file_path}: {str(e)}")


# Gradio 接口函数
def run_model(x_step, y_step, meters, model_type):
    # 清理缓存文件
    clear_cache()

    # 加载参数
    params = load_params()
    selected_features = params.get("selected_features", [])
    SAVED_MODELS_DIR = params.get("SAVED_MODELS_DIR", "./saved_models")

    # 校验必填参数
    if not meters or not y_step:
        return "米数和预测步数 Y 是必填项，请提供完整的输入。", None

    # 转换输入类型
    try:
        y_step = int(y_step)
    except ValueError:
        return "预测步数 Y 必须是整数。", None

    x_step = int(x_step) if x_step and x_step.strip() != "" else None

    # 搜索模型
    selected_models = []

    models_dir = SAVED_MODELS_DIR
    if x_step:
        model_name_pattern = f"use{x_step}step_predict{y_step}step"
    else:
        model_name_pattern = f"predict{y_step}step"

    # 遍历模型目录，查找匹配的模型文件
    for file in os.listdir(models_dir):
        # 检查米数是否匹配
        if f"{model_type}_{meters}m_" not in file:
            continue
        # 检查预测步数是否匹配
        if model_type == "Linear" and file.endswith(".pkl") and model_name_pattern in file:
            selected_models.append(os.path.join(models_dir, file))
        elif model_type in ["LSTM", "AttentionLSTM", "Transformer"] and file.endswith(".pth") and model_name_pattern in file:
            selected_models.append(os.path.join(models_dir, file))

    if not selected_models:
        return "未找到符合条件的模型，请检查输入参数。", None

    # 如果 X_step 留空，随机选择一个模型
    if not x_step and len(selected_models) > 1:
        model_path = random.choice(selected_models)
    else:
        # 选择第一个找到的模型
        model_path = selected_models[0]

    # 提取 SEQ_LENGTH 和 PRED_STEPS 从模型名称
    try:
        parts = os.path.basename(model_path).split('_')
        use_step = [part for part in parts if "use" in part]
        predict_step = [part for part in parts if "predict" in part][0]

        SEQ_LENGTH = int(use_step[0].replace("use", "").replace("step", "")) if use_step else 0
        # 移除文件扩展名后再转换
        predict_step_clean = os.path.splitext(predict_step)[0]
        PRED_STEPS = int(predict_step_clean.replace("predict", "").replace("step", ""))
    except Exception as e:
        return f"无法解析模型名称以获取 SEQ_LENGTH 和 PRED_STEPS: {str(e)}", None

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    X_test, y_test, error = load_test_data(meters, selected_features, SEQ_LENGTH, PRED_STEPS)
    if error:
        return error, None

    # 进行预测
    y_pred, error = predict_and_evaluate(model_path, model_type, X_test, device, SEQ_LENGTH, PRED_STEPS)
    if error:
        return error, None

    # 确保 y_pred 和 y_test 的样本数一致
    if y_pred.shape[0] != y_test.shape[0]:
        return f"预测样本数 {y_pred.shape[0]} 与真实样本数 {y_test.shape[0]} 不一致。", None

    # 根据 PRED_STEPS 处理评估指标和绘图
    metrics = ""
    image_paths = []
    if PRED_STEPS > 1:
        # 对每个步长分别计算评估指标和生成图表
        for step in range(PRED_STEPS):
            try:
                y_true_step = y_test[:, step]
                y_pred_step = y_pred[:, step] if y_pred.ndim > 1 else y_pred

                # 计算评估指标
                mse = mean_squared_error(y_true_step, y_pred_step)
                rmse = math.sqrt(mse)
                mae = mean_absolute_error(y_true_step, y_pred_step)
                r2 = r2_score(y_true_step, y_pred_step)
                metrics += f"### 步长 {step + 1}\n"
                metrics += f"- MSE: {mse:.4f}\n"
                metrics += f"- RMSE: {rmse:.4f}\n"
                metrics += f"- MAE: {mae:.4f}\n"
                metrics += f"- R²: {r2:.4f}\n\n"

                # 生成绘图并保存到临时文件
                plot_title = f"{os.path.basename(model_path)} 预测 vs 真实结果 (步长 {step + 1})"
                img_path = cache_plot_predictions(y_true_step, y_pred_step, plot_title, step + 1)
                image_paths.append(img_path)
            except Exception as e:
                return f"处理步长 {step + 1} 时出错: {str(e)}", None
    else:
        # 单步预测
        try:
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"

            # 生成绘图并保存到临时文件
            plot_title = f"{os.path.basename(model_path)} 预测 vs 真实结果"
            img_path = cache_plot_predictions(y_test, y_pred, plot_title, 1)
            image_paths.append(img_path)
        except Exception as e:
            return f"处理单步预测时出错: {str(e)}", None

    # 返回评估指标和图像文件路径
    return metrics, image_paths


# Gradio 接口布局
with gr.Blocks() as app:
    gr.Markdown("# 模型运行效果展示")

    with gr.Row():
        x_input = gr.Textbox(label="Use X步", placeholder="留空则搜索任意X步")
        y_input = gr.Textbox(label="Predict Y步", placeholder="请输入Y步")

    with gr.Row():
        meters_input = gr.Dropdown(choices=[10, 50, 100], label="高度 (米)", value=10)
        model_type_input = gr.Radio(choices=["Linear", "AttentionLSTM", "LSTM", "Transformer"], label="模型类型", value="Linear")

    run_button = gr.Button("确认")

    with gr.Row():
        metrics_output = gr.Markdown(label="评估指标")

    with gr.Row():
        plots_output = gr.Gallery(label="预测 vs 真实结果")

    run_button.click(
        fn=run_model,
        inputs=[x_input, y_input, meters_input, model_type_input],
        outputs=[metrics_output, plots_output]
    )

if __name__ == "__main__":
    app.launch()
