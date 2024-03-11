import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 定义真实函数
def true_function(x):
    return 2 * x**2 - 3 * x + 1

# 生成带有噪声的训练数据
np.random.seed(0)
torch.manual_seed(0)
num_samples = 100
x_train = np.random.uniform(-5, 5, size=(num_samples, 1)).astype(np.float32)
y_train = true_function(x_train) + np.random.normal(scale=1.0, size=(num_samples, 1)).astype(np.float32)

# 定义ReLU神经网络模型
class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReLUNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义模型参数
input_dim = 1
hidden_dim = 32
model = ReLUNetwork(input_dim, hidden_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 转换数据为Tensor
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

# 训练模型
num_epochs = 500
batch_size = 32
dataset = TensorDataset(x_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在训练数据上进行预测
model.eval()
with torch.no_grad():
    x_values = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y_pred = model(x_values).numpy()

# 绘制结果
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label='Training Data')
plt.plot(x_values.numpy(), true_function(x_values).numpy(), 'r-', label='True Function')
plt.plot(x_values.numpy(), y_pred, 'g--', label='ReLU Network Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting a Function with ReLU Network')
plt.legend()
plt.grid(True)
plt.show()
