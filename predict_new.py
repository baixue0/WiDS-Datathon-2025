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


#--------------- read inputs ---------------
targets = pd.read_excel(config['data_paths']['train_targets'],index_col=0)
cntm = pd.read_csv(config['data_paths']['train_cntm'],index_col=0)
cntm_test = pd.read_csv(config['data_paths']['testnew_cntm'],index_col=0)
meta_qt = pd.read_excel(config['data_paths']['train_meta_qt'],index_col=0)
meta_qt_test = pd.read_excel(config['data_paths']['testnew_meta_qt'],index_col=0)
meta_ct = pd.read_excel(config['data_paths']['train_meta_ct'],index_col=0)
meta_ct_test = pd.read_excel(config['data_paths']['testnew_meta_ct'],index_col=0)
print('loaded')

#--------------- normalize train and test cntm using the same average and std---------------
average = cntm.mean()
std = cntm.std()
cntm, cntm_test = (cntm-average)/std, (cntm_test-average)/std

#--------------- scale meta data ---------------
meta = pd.concat([meta_qt.drop('MRI_Track_Age_at_Scan',axis=1),(meta_ct[['Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Occ']].sum(1)/5).rename('Occ').to_frame()],axis=1)
meta_test = pd.concat([meta_qt_test.drop('MRI_Track_Age_at_Scan',axis=1),(meta_ct_test[['Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Occ']].sum(1)/5).rename('Occ').to_frame()],axis=1)

scaler = RobustScaler()
meta = pd.DataFrame(scaler.fit_transform(meta),index=meta.index,columns=meta.columns).fillna(0)
meta_test = pd.DataFrame(scaler.transform(meta_test),index=meta_test.index,columns=meta_test.columns).fillna(0)

data = {'cntm':cntm_test,'meta':meta_test}

#--------------- predict ---------------

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
preds['ADHD_Outcome'] = (np.stack(preds['ADHD_Outcome']).mean(axis=1)>0.5).astype(int)
preds['Sex_F'] = (np.stack(preds['Sex_F']).mean(axis=1)>0.5).astype(int)
preds = pd.DataFrame(preds,index=data['cntm'].index)
print(preds)
preds.to_csv(config['data_paths']['submission'])
