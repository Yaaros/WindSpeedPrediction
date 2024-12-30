# train.py

import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from utils.log_utils import log  # 引入日志记录函数
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    log('Linear Regression model trained.', level='info')
    return model

def evaluate_model(model, val_loader, criterion, device='cpu'):
    """
    评估模型在验证集上的损失。

    Parameters:
    - model: PyTorch模型。
    - val_loader: 验证集DataLoader。
    - criterion: 损失函数。
    - device: 设备（'cpu'或'cuda'）。

    Returns:
    - avg_val_loss: 平均验证损失。
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, print_every=10):
    """
    通用的PyTorch模型训练函数。每训练一个epoch，输出当前的训练损失和验证损失。
    并加入早停机制：当最近 t 个 epoch 的 Val_Loss 波动小于0.5且 Val_Loss 的最大值小于15，
    或者最近 t 个 epoch 的 R^2 都大于0.85 时，停止训练。

    Parameters:
    - model: PyTorch模型。
    - train_loader: 训练集DataLoader。
    - val_loader: 验证集DataLoader。
    - criterion: 损失函数。
    - optimizer: 优化器。
    - num_epochs: 训练的总轮数。
    - device: 设备（'cpu'或'cuda'）。
    - print_every: 每多少个epoch输出一次损失。

    Returns:
    - model: 训练后的模型。
    - early_stop: 是否提前停止训练的标志（True/False）。
    """
    t = int(round(math.sqrt(num_epochs)))  # 设定 interval
    val_losses_history = []  # 用于存储每个epoch的Val_Loss
    val_r2_history = []      # 用于存储每个epoch的R^2
    early_stop = False        # 早停标志

    # 初始化 ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    model.to(device)
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算该epoch的平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)

        # 计算验证损失
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses_history.append(avg_val_loss)

        # 调度器 step
        scheduler.step(avg_val_loss)

        # 计算 R^2
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
        r2 = r2_score(y_true, y_pred)
        val_r2_history.append(r2)
        model.train()

        # 检查早停条件
        if len(val_losses_history) >= t:
            # 检查 Val_Loss 波动
            recent_val_losses = val_losses_history[-t:]
            mean_val_loss = np.mean(recent_val_losses)
            max_deviation = np.max(recent_val_losses) - mean_val_loss
            min_deviation = mean_val_loss - np.min(recent_val_losses)
            variation = max_deviation + min_deviation
            max_val_loss = np.max(recent_val_losses)

            # 检查 R^2 条件
            recent_r2 = val_r2_history[-t:]
            mean_r2 = np.mean(recent_r2)

            # 这两个数也是完全靠试出来的
            stop_condition_loss2 = (variation < 0.002) and (mean_r2 > 0.5)
            stop_condition_loss = (variation < 0.025) and (mean_r2 > 0.75)
            stop_condition_r2 = (mean_r2 > 0.85)

            if stop_condition_loss or stop_condition_r2 or stop_condition_loss2:
                if stop_condition_r2:
                    log(f'Early stopping triggered at epoch {epoch}: Mean R^2 {mean_r2:.2f} > 0.85', level='warning')
                else:
                    log(f'Early stopping triggered at epoch {epoch}: Variation {variation:.2f} < 0.035', level='warning')
                early_stop = True
                break

        # 打印损失信息
        if epoch % print_every == 0 or epoch == num_epochs:
            log(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, R^2: {r2:.4f}', level='info')

    # 训练完成后，计算 MSE, RMSE, R^2
    model.eval()
    y_true_total = []
    y_pred_total = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            y_true_total.extend(y_batch.cpu().numpy())
            y_pred_total.extend(outputs.cpu().numpy())

    mse = mean_squared_error(y_true_total, y_pred_total)
    rmse = math.sqrt(mse)
    r2_final = r2_score(y_true_total, y_pred_total)

    log(f'Training completed. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2_final:.4f}', level='info')

    return model, early_stop
