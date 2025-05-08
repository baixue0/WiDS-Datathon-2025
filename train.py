import os
import json
import copy
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
from autoencoder import Autoencoder

with open('config/SETTINGS.json') as f:
    config = json.load(f)
print(config)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_col", help='ADHD_Outcome or Sex_F', required=True)
parser.add_argument("--seed", help='0, 1, 2', required=True)
args = parser.parse_args()
target_col = args.target_col
seed = int(args.seed)-1

data = np.load(os.path.join(config['data_paths']['clean_data_dir'],'train_val'+str(seed)+'.npy'),allow_pickle=True).item()
print(data.keys())

best_scores = []
best_models = []

for class_weight in [0.1,1,10]:
    hgb = HistGradientBoostingClassifier(class_weight={0:1,1:class_weight})
    hgb.fit(pd.concat([data['cntm_train'],data['meta_train']],axis=1), data['targets_train'][target_col])
    pred = hgb.predict(pd.concat([data['cntm_val'],data['meta_val']],axis=1))
    best_score = f1_score(data['targets_val'][target_col],pred)
    print('hgb',class_weight,round(best_score,3))
    best_scores.append(best_score)
    best_models.append(('hgb',hgb))

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(data['cntm_train'].to_numpy()),
    torch.tensor(data['targets_train'][target_col].to_numpy()),
    torch.tensor(data['meta_train'].to_numpy()),
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(data['cntm_val'].to_numpy()),
    torch.tensor(data['targets_val'][target_col].to_numpy()),
    torch.tensor(data['meta_val'].to_numpy()),
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ae = Autoencoder(data['cntm_train'].shape[1],data['meta_train'].shape[1],config['latent_dim'],1, dropout=config['dropout']).to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
loss_mse = torch.nn.MSELoss(reduction="mean")
loss_bce = torch.nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model_wts = None
best_score,best_std = -float('inf'),-float('inf')
for epoch in range(1, config['epochs'] + 1):
    train_loss = ae.train_model(epoch, train_loader, optimizer, loss_mse, loss_bce, device)
    encoded_val,xs_val,recs_val,target_val,metas_val,clses_val = ae.validate_model(epoch, val_loader, device)
    pred = (clses_val>0.5).astype(int)
    f1 = f1_score(target_val,pred)
    print(epoch,round(f1,6),round(pred.std(),6))
    if (f1 > best_score) and (pred.std()>0.4):
        best_score = f1
        best_std = pred.std()
        best_model_wts = copy.deepcopy(ae.state_dict())
print('ae',round(best_score,3))
if best_model_wts is not None:
    best_scores.append(best_score)
    best_models.append(('ae',best_model_wts))

os.makedirs(config['data_paths']['model_dir'], exist_ok=True)
modeltype,model = best_models[np.argmax(best_scores)]
if modeltype=='ae':
    torch.save(best_model_wts, os.path.join(config['data_paths']['model_dir'], f'{target_col}|{seed}.pth'))
else:
    with open(os.path.join(config['data_paths']['model_dir'], f'{target_col}|{seed}.pkl'), 'wb') as f:
        pickle.dump(model, f)
