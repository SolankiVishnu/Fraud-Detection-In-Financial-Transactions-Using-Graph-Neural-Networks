import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
import pickle  # Import pickle

df = pd.read_csv('data/transactions.csv')  # <- UPDATED path
df.columns = df.columns.str.strip()

user_encoder = LabelEncoder()
df['User_ID'] = user_encoder.fit_transform(df['User_ID'])
num_nodes = df['User_ID'].nunique()

categorical_cols = [
    'Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category',
    'Authentication_Method', 'Card_Type'
]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

feature_cols = [
    'Transaction_Amount', 'Account_Balance', 'Device_Type', 'Location',
    'Merchant_Category', 'IP_Address_Flag', 'Previous_Fraudulent_Activity',
    'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d',
    'Failed_Transaction_Count_7d', 'Card_Type', 'Card_Age',
    'Transaction_Distance', 'Authentication_Method', 'Risk_Score', 'Is_Weekend'
]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

user_features = pd.DataFrame(scaled_features, columns=feature_cols)
user_features['User_ID'] = df['User_ID']
node_features = user_features.groupby('User_ID').mean().sort_index()
x = torch.tensor(node_features.values, dtype=torch.float)

senders = df['User_ID'].values
receivers = df.sample(frac=1.0)['User_ID'].values
edge_index = torch.tensor([senders, receivers], dtype=torch.long)

user_labels = df.groupby('User_ID')['Fraud_Label'].max().sort_index()
y = torch.tensor(user_labels.values, dtype=torch.long)

train_idx, test_idx = train_test_split(range(num_nodes), test_size=0.3, stratify=y, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data = Data(x=x, edge_index=edge_index, y=y)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=x.shape[1], hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch:02d} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/gnn_model.pt")
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

model.eval()
pred = model(data).argmax(dim=1)
print("\n📊 Classification Report:")
print(classification_report(data.y[test_mask], pred[test_mask]))

G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i, fraud=bool(y[i].item()))

edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
G.add_edges_from(edge_list)
color_map = ['red' if y[node] == 1 else 'green' for node in G.nodes]

plt.figure(figsize=(12, 10))
subgraph = G.subgraph(list(G.nodes)[:100])
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(subgraph, pos, node_color=color_map[:100], node_size=50, with_labels=False, alpha=0.8, edge_color='gray')
plt.title("📍 Transaction Graph (Red = Fraudulent Users)")
plt.axis('off')
plt.show()
