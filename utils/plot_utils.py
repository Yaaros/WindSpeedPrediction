# utils/plot_utils.py
import logging
import os

import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, model_name, save_dir='plots'):
    """
    绘制预测结果与实际结果的折线图。

    Parameters:
    - y_true (np.ndarray): Actual values.
    - y_pred (np.ndarray): Predicted values.
    - model_name (str): Model name, used for naming the chart.
    - save_dir (str): Directory to save the chart.

    Returns:
    - None
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{model_name}_predictions.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Plot saved to {plot_path}")
# 将 Matplotlib 图表保存到临时文件并返回路径
def cache_plot_predictions(y_true, y_pred, title, step):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    # 设置中文字体和显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 保存图像到 cache 目录
    temp_file_path = f'./cache/temp{step}.png'
    plt.savefig(temp_file_path)
    plt.close()

    return temp_file_path