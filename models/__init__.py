# ./model/__init__.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

# 线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# LSTM 多步预测模型
class LSTMModelMultistep(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, m_steps):
        super(LSTMModelMultistep, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.m_steps = m_steps

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, m_steps)  # 输出多个步长

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, n_steps, hidden_size)
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, m_steps)
        return out

# Attention LSTM 模型
class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, m_steps, n_heads=4):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.m_steps = m_steps

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, n_heads)
        self.fc = nn.Linear(hidden_size, m_steps)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch_size, n_steps, hidden_size)
        lstm_out = lstm_out.permute(1, 0, 2)  # (n_steps, batch_size, hidden_size)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (n_steps, batch_size, hidden_size)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, n_steps, hidden_size)
        out = attn_output[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, m_steps)
        return out

# Transformer 多步预测模型
class TemporalFusionTransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, m_steps):
        super(TemporalFusionTransformerModel, self).__init__()
        self.d_model = d_model
        self.m_steps = m_steps

        self.input_linear = nn.Linear(input_size, d_model)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc_out = nn.Linear(d_model, m_steps)

    def forward(self, src):
        src = self.input_linear(src) * (self.d_model ** 0.5)  # (batch_size, n_steps, d_model)
        src = src.permute(1, 0, 2)  # (n_steps, batch_size, d_model)

        tgt = torch.zeros(self.m_steps, src.size(1), self.d_model).to(src.device)  # (m_steps, batch_size, d_model)
        out = self.transformer(src, tgt)  # (m_steps, batch_size, d_model)
        out = out.permute(1, 0, 2)  # (batch_size, m_steps, d_model)
        out = self.fc_out(out[:, -1, :])  # (batch_size, m_steps)
        return out