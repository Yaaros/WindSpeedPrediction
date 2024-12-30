# data.py

from torch.utils.data import TensorDataset, DataLoader

from utils.path_str import get_target_directory


def handle_outliers(df, columns, factor=3):
    """
    处理异常值，采用3倍标准差法，将异常值替换为均值。

    参数:
    - df (pd.DataFrame): 输入的数据框。
    - columns (list): 需要处理异常值的列名列表。
    - factor (int, optional): 标准差倍数，默认3。

    返回:
    - pd.DataFrame: 处理后的数据框。
    """
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
        df[col] = np.where(
            (df[col] < lower_bound) | (df[col] > upper_bound),
            mean,
            df[col]
        )
    return df

# load_and_preprocess_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(dataset_type, selected_features):
    """
    加载并预处理指定类型的数据集。

    参数:
    - dataset_type (str): 数据集类型，如 '10m', '50m', '100m'。
    - selected_features (list): 选择的特征列列表。

    返回:
    - X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    splits = {
        'train': f'train-windspeed-of-{dataset_type}.parquet',
        'val': f'val-windspeed-of-{dataset_type}.parquet',
        'test': f'test-windspeed-of-{dataset_type}.parquet'
    }
    try:
        # 尝试从本地目录加载数据
        train_path = get_target_directory(splits['train'])
        val_path = get_target_directory(splits['val'])
        test_path = get_target_directory(splits['test'])

        train_dataset = pd.read_parquet(train_path)
        val_dataset = pd.read_parquet(val_path)
        test_dataset = pd.read_parquet(test_path)

        print("成功从本地加载训练、验证和测试数据。")

        X_test, X_train, X_val, y_test, y_train, y_val = preprocess(dataset_type, selected_features, test_dataset,
                                                                    train_dataset, val_dataset)
    except FileNotFoundError as e:
        print(f"本地数据加载失败: {e}")
        print("尝试从网络资源加载数据。")
        try:
            # 从网络资源加载训练数据
            network_train_path = f"hf://datasets/Antajitters/WindSpeed_{dataset_type}/" + splits["train"]
            network_val_path = f"hf://datasets/Antajitters/WindSpeed_{dataset_type}/" + splits["val"]
            network_test_path = f"hf://datasets/Antajitters/WindSpeed_{dataset_type}/" + splits["test"]

            train_dataset = pd.read_parquet(network_train_path)
            val_dataset = pd.read_parquet(network_val_path)
            test_dataset = pd.read_parquet(network_test_path)

            print("成功从网络加载训练、验证和测试数据。")
            X_test, X_train, X_val, y_test, y_train, y_val = preprocess(dataset_type, selected_features, test_dataset,
                                                                        train_dataset, val_dataset)
        except Exception as network_e:
            raise RuntimeError(f"从网络加载数据失败: {network_e}")

    # 特征缩放（标准化）
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(dataset_type, selected_features, test_dataset, train_dataset, val_dataset):
    # 处理缺失值
    numeric_columns = train_dataset.select_dtypes(include=[np.number]).columns
    train_dataset[numeric_columns] = train_dataset[numeric_columns].fillna(train_dataset[numeric_columns].mean())
    val_dataset[numeric_columns] = val_dataset[numeric_columns].fillna(val_dataset[numeric_columns].mean())
    test_dataset[numeric_columns] = test_dataset[numeric_columns].fillna(test_dataset[numeric_columns].mean())
    # 处理异常值
    train_dataset = handle_outliers(train_dataset,
                                    ['SpeedAvg', 'DirectionAvg', 'TemperatureAvg', 'PressureAvg', 'HumidtyAvg'])
    val_dataset = handle_outliers(val_dataset,
                                  ['SpeedAvg', 'DirectionAvg', 'TemperatureAvg', 'PressureAvg', 'HumidtyAvg'])
    test_dataset = handle_outliers(test_dataset,
                                   ['SpeedAvg', 'DirectionAvg', 'TemperatureAvg', 'PressureAvg', 'HumidtyAvg'])
    # 特征工程：计算压力差、温度差和湿度差
    for df in [train_dataset, val_dataset, test_dataset]:
        df['PressureDelta'] = df['PressureMax'] - df['PressureAvg']
        df['TemperatureDelta'] = df['TemperatureMax'] - df['TemperatureAvg']
        df['HumidtyDelta'] = df['HumityMax'] - df['HumidtyAvg']
        df['height'] = int(dataset_type[:-1])
    # 特征选择
    available_features, missing_features = select_available_features(train_dataset, selected_features)
    if missing_features:
        print(f"警告: 以下选择的特征在数据集中未找到: {missing_features}")
    available_features.remove('SpeedAvg')
    if(int(dataset_type[:-1])!=10):
        available_features.append('Speed Avg 10m')
    # 提取特征和目标
    X_train = train_dataset[available_features].values
    y_train = train_dataset['SpeedAvg'].values
    X_val = val_dataset[available_features].values
    y_val = val_dataset['SpeedAvg'].values
    X_test = test_dataset[available_features].values
    y_test = test_dataset['SpeedAvg'].values
    return X_test, X_train, X_val, y_test, y_train, y_val


def select_available_features(df, feature_columns):
    """
    选择存在于数据框中的特征列。

    参数:
    - df (pd.DataFrame): 输入的数据框。
    - feature_columns (list): 预定义的特征列列表。

    返回:
    - available_features (list): 实际存在的特征列列表。
    - missing_features (list): 缺失的特征列列表。
    """
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"警告: 以下特征在数据集中缺失，将被忽略: {missing_features}")
    return available_features, missing_features


def create_sequences(features, targets, n_steps, m_steps):
    """
    根据时间步创建输入序列和目标序列。

    参数:
    - features (np.ndarray): 特征数组，形状为 (num_samples, num_features)。
    - targets (np.ndarray): 目标数组，形状为 (num_samples,)。
    - n_steps (int): 使用的历史时间步数。
    - m_steps (int): 预测的未来时间步数。

    返回:
    - X (np.ndarray): 输入特征序列，形状为 (num_sequences, n_steps, num_features)。
    - Y (np.ndarray): 目标序列，形状为 (num_sequences, m_steps)。
    """
    X, Y = [], []
    for i in range(len(features) - n_steps - m_steps + 1):
        X_seq = features[i:i + n_steps]
        Y_seq = targets[i + n_steps:i + n_steps + m_steps]
        if len(Y_seq) == m_steps:
            if m_steps == 1:
                Y_seq = Y_seq[0]  # 将单步预测转换为标量
                Y.append([Y_seq])  # 保持二维结构
            else:
                Y.append(Y_seq.tolist())  # 多步预测保留列表结构
            X.append(X_seq)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def prepare_dataloaders(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor,
                        batch_size=64):
    """
    创建训练、验证和测试的 DataLoader。

    参数:
    - X_train_tensor, y_train_tensor: 训练数据和目标。
    - X_val_tensor, y_val_tensor: 验证数据和目标。
    - X_test_tensor, y_test_tensor: 测试数据和目标。
    - batch_size (int): 批次大小。

    返回:
    - train_loader, val_loader, test_loader: 分别为训练、验证和测试的 DataLoader。
    """
    # 创建 TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
