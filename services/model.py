import torch
import torch.nn as nn


class LSTMModel(nn.Module): # type: ignore
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    # 输出：一个预测值（0 - 1 之间，表示上涨概率）
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cpu')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cpu')
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class TradingStrategy:
    def __init__(self, initial_balance=1000000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.current_shares = 0
        self.threshold = 0.001
        self.profits = []  # 用于记录每笔交易的盈亏

    def execute_trades(self, prediction, current_price, current_date):
        try:
            # 计算信号
            signal = prediction - 0.5

            # 买入信号
            if signal > self.threshold and self.position <= 0:
                shares_to_buy = int(self.balance * 0.95 / current_price)
                print(f"买入 : {shares_to_buy}股 @ {current_price:.2f}")
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    if cost <= self.balance:
                        self.balance -= cost
                        self.current_shares = shares_to_buy
                        self.position = 1
                        self.trades.append({
                            'date': current_date,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'cost': cost,
                            'balance': self.balance
                        })
                        print(f"执行买入 : {shares_to_buy}股 @ {current_price:.2f}")
                        print(f"交易成本 : {cost:.2f}")
                        print(f"剩余余额 : {self.balance:.2f}")

            # 卖出信号
            elif signal < -self.threshold and self.position >= 0:
                if self.current_shares > 0:
                    revenue = self.current_shares * current_price
                    profit = revenue - self.trades[-1]['cost']  # 计算盈亏
                    self.profits.append(profit)
                    self.balance += revenue
                    self.trades.append({
                        'date': current_date,
                        'type': 'SELL',
                        'price': current_price,
                        'shares': self.current_shares,
                        'revenue': revenue,
                        'profit': profit,
                        'balance': self.balance
                    })
                    print(f"执行卖出 : {self.current_shares}股 @ {current_price:.2f}")
                    print(f"交易收入 : {revenue:.2f}")
                    print(f"交易盈亏 : {profit:.2f}")
                    print(f"当前余额 : {self.balance:.2f}")
                    self.current_shares = 0
                    self.position = -1

        except Exception as e:
            print(f"交易执行错误 : {str(e)}")
            import traceback
            print(traceback.format_exc())

    def get_performance_metrics(self):
        if not self.trades:
            return {
                'total_profit': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'final_balance': self.balance,
                'return_rate': 0.0
            }

        num_trades = len(self.profits)
        total_profit = sum(self.profits) if self.profits else 0
        win_trades = len([p for p in self.profits if p > 0]) if self.profits else 0
        
        win_rate = (win_trades / num_trades) if num_trades > 0 else 0
        avg_profit = (total_profit / num_trades) if num_trades > 0 else 0
        return_rate = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        return {
            'total_profit': total_profit,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'final_balance': self.balance,
            'return_rate': return_rate
        }
    
def train_model(model, X_train, y_train, epochs, learning_rate, batch_size):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X_train) // batch_size)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses
