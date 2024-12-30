# main.py

import os
import json
from datetime import datetime

import joblib
import torch
from torch import optim

from data import load_and_preprocess_data, prepare_dataloaders, create_sequences
from evaluate import compute_metrics
from models import (
    LinearRegressionModel,
    LSTMModelMultistep,
    AttentionLSTMModel,
    TemporalFusionTransformerModel,
)
from train import train_model, train_linear
from utils.log_utils import log, setup_logging
from utils.plot_utils import plot_predictions


def load_params(filepath='Params.json'):
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params


def main():
    # 加载参数
    params = load_params()

    NUM_EPOCHS = params["NUM_EPOCHS"]
    BATCH_SIZE = params["BATCH_SIZE"]
    SAVED_MODELS_DIR = params["SAVED_MODELS_DIR"]
    PLOTS_DIR = params["PLOTS_DIR"]
    selected_features = params["selected_features"]
    dataset_types = params["dataset_types"]
    PREDICTS = params["PREDICTS"]  # List of [SEQ_LENGTH, PRED_LENGTH]

    # 初始化日志记录
    setup_logging()

    log("=== 训练开始 ===\n", level='info')

    # 初始化保存模型和图表的目录
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 遍历每个数据集类型
    for dataset_type in dataset_types:
        log(f"===== 处理数据集: {dataset_type} =====", level='info')

        # 遍历每个预测配置
        for idx, (SEQ_LENGTH, PRED_STEPS) in enumerate(PREDICTS):
            log(f"--- 开始训练：SEQ_LENGTH={SEQ_LENGTH}, PRED_STEPS={PRED_STEPS} ---", level='info')

            # 1. 加载和预处理数据
            X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
                dataset_type=dataset_type,
                selected_features=selected_features
            )

            # 打印选择的特征
            log(f"Selected Features: {selected_features}", level='info')

            # 2. 创建序列数据
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH, PRED_STEPS)
            X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LENGTH, PRED_STEPS)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH, PRED_STEPS)

            # 打印数据形状
            log(f"X_train_seq shape: {X_train_seq.shape}", level='info')  # (num_sequences, n_steps, num_features)
            log(f"y_train_seq shape: {y_train_seq.shape}", level='info')  # (num_sequences, m_steps)
            log(f"X_val_seq shape: {X_val_seq.shape}", level='info')
            log(f"y_val_seq shape: {y_val_seq.shape}", level='info')
            log(f"X_test_seq shape: {X_test_seq.shape}", level='info')
            log(f"y_test_seq shape: {y_test_seq.shape}", level='info')

            # 转换为 Torch 张量
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)  # (num_sequences, m_steps)
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)  # (num_sequences, m_steps)
            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)  # (num_sequences, m_steps)

            # 验证张量形状
            log(f"y_train_tensor shape: {y_train_tensor.shape}", level='info')  # (num_sequences, m_steps)
            log(f"y_val_tensor shape: {y_val_tensor.shape}", level='info')
            log(f"y_test_tensor shape: {y_test_tensor.shape}", level='info')

            # 3. 创建 DataLoader
            train_loader, val_loader, test_loader = prepare_dataloaders(
                X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, BATCH_SIZE
            )

            # 4. 定义和初始化模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            log(f"使用设备: {device}", level='info')

            # 初始化模型实例
            lstm_model = LSTMModelMultistep(
                input_size=X_train_tensor.shape[2],  # num_features
                hidden_size=128,
                num_layers=2,
                m_steps=PRED_STEPS
            ).to(device)

            attention_lstm_model = AttentionLSTMModel(
                input_size=X_train_tensor.shape[2],
                hidden_size=128,
                num_layers=2,
                m_steps=PRED_STEPS,
                n_heads=4
            ).to(device)

            transformer_model = TemporalFusionTransformerModel(
                input_size=X_train_tensor.shape[2],
                d_model=32,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=256,
                m_steps=PRED_STEPS
            ).to(device)
            # 定义优化器
            lstm_optimizer = optim.AdamW(
                lstm_model.parameters(),
                lr=5e-6,  # 调整学习率
            )

            attention_lstm_optimizer = optim.Adam(
                attention_lstm_model.parameters(),
                lr=5e-6  # 调整学习率
            )

            transformer_optimizer = optim.AdamW(
                transformer_model.parameters(),
                lr=5e-6,
            )
            # 定义模型字典
            models = {
                'Linear': {
                    'type': 'linear',
                    'train_function': 'train_linear'
                },
                'LSTM': {
                    'type': 'pytorch',
                    'model': lstm_model,
                    'train_function': 'train_model',
                    'optimizer': lstm_optimizer
                },
                'AttentionLSTM': {
                    'type': 'pytorch',
                    'model': attention_lstm_model,
                    'train_function': 'train_model',
                    'optimizer': attention_lstm_optimizer
                },
                'TemporalFusionTransformer': {
                    'type': 'pytorch',
                    'model': transformer_model,
                    'train_function': 'train_model',
                    'optimizer': transformer_optimizer
                },
            }

            # 初始化一个字典来存储训练后的模型
            trained_models = {}

            # 遍历模型字典并训练
            for model_name, model_info in models.items():
                log(f"=====\n训练：{model_name} 对于数据集: {dataset_type}, SEQ_LENGTH={SEQ_LENGTH}, PRED_STEPS={PRED_STEPS}\n========",
                    level='info')
                if model_info['type'] == 'linear':
                    # 训练线性回归模型
                    linear_model = train_linear(X_train_tensor.numpy(), y_train_seq)
                    # 保存模型
                    model_save_path = os.path.join(
                        SAVED_MODELS_DIR,
                        f"{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step.pkl"
                    )
                    joblib.dump(linear_model, model_save_path)
                    log(f"Linear Regression model saved to {model_save_path}", level='info')
                    # 存储到 trained_models
                    trained_models[model_name] = linear_model
                    log("===== 训练完毕 =====\n", level='info')
                else:
                    # 训练 PyTorch 模型
                    model = model_info['model']
                    optimizer = model_info['optimizer']

                    # 训练模型，并获取是否提前停止的标志
                    trained_model, early_stop = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        criterion=torch.nn.MSELoss(),
                        optimizer=optimizer,
                        num_epochs=NUM_EPOCHS,
                        device=device,
                        print_every=10
                    )
                    # 存储到 trained_models
                    trained_models[model_name] = trained_model
                    model_save_path = os.path.join(
                        SAVED_MODELS_DIR,
                        f"{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step.pth"
                    )
                    torch.save(trained_model.state_dict(), model_save_path)

                    log(f"{model_name} model saved to {model_save_path}", level='info')

                    log("===== 训练完毕 =====\n", level='info')

            log(f"=== 训练结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n", level='info')

            # 5. 评估模型
            log("=== 模型评估开始 ===\n", level='info')
            for model_name, model in trained_models.items():
                log(f"=== 评估模型：{model_name} 对于数据集: {dataset_type}, SEQ_LENGTH={SEQ_LENGTH}, PRED_STEPS={PRED_STEPS} ===",
                    level='info')
                if models[model_name]['type'] == 'linear':
                    # 线性回归预测
                    y_pred_linear = model.predict(X_test_tensor.numpy().reshape(X_test_tensor.shape[0], -1))
                    compute_metrics(y_test_seq, y_pred_linear,
                                    f"{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step")
                    # 绘制预测结果
                    plot_predictions(y_test_seq, y_pred_linear,
                                     f"{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step",
                                     save_dir=PLOTS_DIR)
                    log(f"Plot saved to {PLOTS_DIR}/{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step_predictions.png",
                        level='info')
                else:
                    # PyTorch 模型预测
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(X_test_tensor.to(device))
                    y_pred = y_pred.cpu().numpy()
                    compute_metrics(y_test_seq, y_pred,
                                    f"{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step")
                    # 绘制预测结果
                    plot_predictions(y_test_seq, y_pred,
                                     f"{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step",
                                     save_dir=PLOTS_DIR)
                    log(f"Plot saved to {PLOTS_DIR}/{model_name}_{dataset_type}_use{SEQ_LENGTH}step_predict{PRED_STEPS}step_predictions.png",
                        level='info')
                    log("\n", level='info')
                    log("=== 模型评估结束 ===\n", level='info')


if __name__ == "__main__":
    main()
