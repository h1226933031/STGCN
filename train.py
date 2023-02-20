import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from stgcn import *
import matplotlib.pyplot as plt
import time

torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# File Path
matrix_path = "./dataset/adjacency_matrix_v3.csv"
data_path = "./dataset/data_per_10min.csv"
save_path = "./save/wyh_model.pt"

# Hyper-parameter
day_slot = 105
n_train, n_val, n_test = 18, 2, 2  # num of days in train_val_test
n_his = 12
n_pred = 3
n_route = 288
Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0
batch_size = 50
epochs = 50
lr = 1e-4

# Graph Parameter
W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)  # Ks(kernel size) x 288 x 288
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)

# Standardization & Transform Data
train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)  # 加载
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)  # fit_transform(X_train)
x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

# Data Loader
train_data = torch.utils.data.TensorDataset(x_train, y_train)  # tensor数据包装成dataset
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

# Loss & Model & Optimizer
loss = nn.MSELoss()
model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

# LR Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# training
min_val_loss = np.inf
train_loss_list, validation_loss_list = [], []

tic = time.time()
for epoch in range(epochs):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]  # item()返回的是一个浮点型数据
        n += y.shape[0]  # 每次加50(batch_size)
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    train_loss_list.append(l_sum / n)
    validation_loss_list.append(val_loss)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
toc = time.time()
print("训练用时", toc - tic, 's')

# visualization: loss by epoches
plt.rcParams['font.sans-serif'] = 'STSong'
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(1, epochs + 1), train_loss_list, label='train loss')
ax.plot(range(1, epochs + 1), validation_loss_list, label='validation loss')
ax.set_xlabel("epochs", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# ax.set_xticklabels(labels = range(1, epochs + 1), fontsize = 15,rotation=45)
ax.set_title("STGCN训练性能", fontsize=30)
ax.legend(fontsize=25)
plt.show()

# evaluate
evaluate_loss = pd.DataFrame(train_loss_list, columns=['STGCN-train loss'], index=range(1, epochs + 1))
evaluate_loss = pd.concat(
    [evaluate_loss, pd.DataFrame(validation_loss_list, columns=['STGCN-validation loss'], index=range(1, epochs + 1))],
    axis=1)
evaluate_loss.to_csv('STGCN_evaluation.csv', mode='a', header=True, index=True)

# load best model and evaluate
best_model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
best_model.load_state_dict(torch.load(save_path))
l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
