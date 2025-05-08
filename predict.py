import os
import json
import copy
import numpy as np
import pandas as pd
import pickle
import glob

import torch
from torch.utils.data import DataLoader
from autoencoder import Autoencoder

with open('config/SETTINGS.json') as f:
    config = json.load(f)
print(config)

data = np.load(os.path.join(config['data_paths']['clean_data_dir'],'test.npy'),allow_pickle=True).item()

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(data['cntm'].to_numpy()),
    torch.tensor(data['meta'].to_numpy())
)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_paths = os.listdir(config['data_paths']['model_dir'])
print(model_paths)
preds = {'ADHD_Outcome':[],'Sex_F':[]}
for path in model_paths:
    if path.endswith('.pth'):
        model = Autoencoder(data['cntm'].shape[1],data['meta'].shape[1],config['latent_dim'],1, dropout=config['dropout']).to(device)
        wts = torch.load(os.path.join(config['data_paths']['model_dir'],path))
        model.load_state_dict(wts)
        pred = model.predict(test_loader, device).squeeze()
    if path.endswith('.pkl'):
        with open((os.path.join(config['data_paths']['model_dir'],path)), 'rb') as f:
            model = pickle.load(f)
            pred = model.predict(pd.concat([data['cntm'],data['meta']],axis=1)).squeeze()
    col = path.split('|')[0]
    preds[col].append(pred)
    print(path,pred)
preds = pd.concat([
    pd.Series((np.stack(preds['ADHD_Outcome']).mean(axis=1)>0.5).astype(int), index=data['cntm'].index).rename('ADHD_Outcome'),
    pd.Series((np.stack(preds['Sex_F']).mean(axis=1)>0.5).astype(int), index=data['cntm'].index).rename('Sex_F'),
],axis=1)
print(preds)
preds.to_csv(config['data_paths']['submission'])
