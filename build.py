from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchmetrics import Accuracy
import pandas as pd
import numpy as np
import torch.nn.functional as F



# import data
SEED = 50
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data1['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
mappings = {
   "setosa": 0,
   "versicolor": 1,
   "virginica": 2
}
data1["species"] = data1["species"].apply(lambda x: mappings[x])



# build model
class IrisModelv1(nn.Module):
    def __init__(self, input_size, hidden, out):                
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_size, out_features=hidden)
        self.layer2 = nn.Linear(in_features=hidden, out_features=hidden)
        self.layer3 = nn.Linear(in_features=hidden, out_features=out)
    def forward(self, x):
        return self.layer3(F.relu(self.layer2(F.relu(self.layer1(x)))))

model_1 = IrisModelv1(4, 10, 3)
loss_fn = nn.CrossEntropyLoss()
optimizer_AD = torch.optim.Adam(params=model_1.parameters(), lr=0.1)
accuracy = Accuracy(task="multiclass", num_classes=3)
