import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from random import seed, randrange

# Load data from CSV file
csv_file_path = 'Vehicle Price.csv'  # Update this path if the file is not in the same directory
df = pd.read_csv(csv_file_path, quotechar='"', escapechar='\\', engine='python')

# Preprocess
df = df.dropna()
df = df.drop(['name', 'description'], axis=1)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

X = df.drop('price', axis=1)
y = df['price']

# Manual split (no sklearn)
seed(42)
indices = np.arange(len(y))
np.random.shuffle(indices)
train_size = int(0.8 * len(y))
train_idx = indices[:train_size]
test_idx = indices[train_size:]
X_train = X.iloc[train_idx].values
y_train = y.iloc[train_idx].values
X_test = X.iloc[test_idx].values
y_test = y.iloc[test_idx].values

# Decision Tree from scratch (adapted for regression with MSE)
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def mse(groups):
    n = sum(len(group) for group in groups)
    score = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        y_group = [row[-1] for row in group]
        mean = sum(y_group) / size
        score += sum((yi - mean)**2 for yi in y_group) * (size / n)
    return score

def get_split(dataset):
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            score = mse(groups)
            if score < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], score, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return sum(outcomes) / len(outcomes) if len(outcomes) > 0 else 0

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

def build_tree(train, max_depth, min_size):
    train_list = train.tolist()
    root = get_split(train_list)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Random Forest from scratch
def bootstrap_sample(dataset):
    sample = []
    n_sample = len(dataset)
    for _ in range(n_sample):
        index = randrange(n_sample)
        sample.append(dataset[index])
    return np.array(sample)

def random_forest(train, n_trees, max_depth, min_size):
    trees = []
    for _ in range(n_trees):
        sample = bootstrap_sample(train)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    return trees

def rf_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return sum(predictions) / len(predictions)

# Neural Network with PyTorch
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Normalize for NN
X_train_nn = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-7)
X_test_nn = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-7)  # Use train stats

y_train_nn = y_train.reshape(-1, 1)
y_test_nn = y_test.reshape(-1, 1)

X_train_t = torch.tensor(X_train_nn, dtype=torch.float32)
y_train_t = torch.tensor(y_train_nn, dtype=torch.float32)
X_test_t = torch.tensor(X_test_nn, dtype=torch.float32)
y_test_t = torch.tensor(y_test_nn, dtype=torch.float32)

model = Net(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_nn = model(X_test_t).numpy().flatten()

# Train DT
train_dt = np.hstack((X_train, y_train.reshape(-1, 1)))
tree = build_tree(train_dt, max_depth=5, min_size=2)
y_pred_dt = [predict(tree, row) for row in X_test]

# Train RF
train_rf = train_dt
trees = random_forest(train_rf, n_trees=10, max_depth=5, min_size=2)
y_pred_rf = [rf_predict(trees, row) for row in X_test]

# Metrics function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

# Results
print("Decision Tree - RMSE:", rmse(y_test, y_pred_dt), "R2:", r2(y_test, y_pred_dt))
print("Random Forest - RMSE:", rmse(y_test, y_pred_rf), "R2:", r2(y_test, y_pred_rf))
print("Neural Network - RMSE:", rmse(y_test, y_pred_nn), "R2:", r2(y_test, y_pred_nn))