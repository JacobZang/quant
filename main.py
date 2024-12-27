import services.get_data as get_data
import services.process_data as process_data
import torch
import torch.nn as nn

# df_shenwan = get_data.get_shenwan_data()
# df_shenwan = process_data.process_data(df_shenwan)

df_hs = get_data.get_hs_data("20230101","20241225")
df_hs = process_data.process_data(df_hs)
print(df_hs.head())
class LSTMModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(LSTMModel, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		out, _ = self.lstm(x)
		out = out[:, -1, :]
		out = self.fc(out)
		return out

# 超参数设置
seq_length = 5
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
batch_size = 32
learning_rate = 0.001
epochs = 20


# 初始化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

