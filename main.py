import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import services.get_data as get_data
import services.process_data as process_data
import services.create_dataset as dataset
import services.model as models
import torch
import torch.nn as nn


df_hs_train = get_data.get_hs_data("20230101","20240101")
df_hs_train = process_data.process_data(df_hs_train)
df_hs_test = get_data.get_hs_data("20240101","20241201")
df_hs_test = process_data.process_data(df_hs_test)

seq_length = 30
input_size = 10
hidden_size = 64
num_layers = 3
output_size = 1
batch_size = 32
learning_rate = 0.001
epochs = 2000


features = ["枢轴点", "第一阻力位", "第一支撑位", "第二阻力位", "第二支撑位", "第三阻力位", "第三支撑位", "收盘价", "最高价", "最低价"]
scaler = MinMaxScaler()
scaled_values_train = scaler.fit_transform(df_hs_train[features])
scaled_values_test = scaler.transform(df_hs_test[features])
df_scaled_train = pd.DataFrame(scaled_values_train, columns=features, index=df_hs_train.index)
df_scaled_test = pd.DataFrame(scaled_values_test, columns=features, index=df_hs_test.index)

# train model
model = models.LSTMModel(input_size, hidden_size, num_layers, output_size)
X, y = dataset.create_dataset(df_scaled_train, window_size=seq_length)
X_train_tensor = torch.FloatTensor(X)
y_train_tensor = torch.FloatTensor(y).view(-1, 1)
train_losses = models.train_model(model, X_train_tensor, y_train_tensor, epochs, learning_rate, batch_size)

strategy = models.TradingStrategy(initial_balance=100000)
model.eval()

# test model
with torch.no_grad():
    for i in range(seq_length, len(df_scaled_test)):
        window = df_scaled_test.iloc[i-seq_length:i].values
        window_tensor = torch.FloatTensor(window).unsqueeze(0)
        prediction = model(window_tensor)
        
        current_price = df_hs_test.iloc[i]['收盘价']
        current_date = df_hs_test.index[i]
        strategy.execute_trades(prediction.item(), current_price, current_date)


performance = strategy.get_performance_metrics()
print("\n=== 交易表现 ===")
print(f"总盈利 : ¥{performance['total_profit']:.2f}")
print(f"交易次数 : {performance['num_trades']}")
print(f"胜率 : {performance['win_rate']*100:.2f}%")
print(f"平均每笔交易盈利 : ¥{performance['avg_profit']:.2f}")
print(f"最终余额 : ¥{performance['final_balance']:.2f}")
print(f"收益率 : {performance['return_rate']:.2f}%")